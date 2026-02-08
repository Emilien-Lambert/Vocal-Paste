import json
import re
from pathlib import Path

import mlx.core as mx
from mlx import nn
from huggingface_hub import snapshot_download

from .model import VoxtralRealtime


def download_model(model_id: str) -> Path:
    path = snapshot_download(
        model_id,
        allow_patterns=[
            "consolidated.safetensors",
            "model*.safetensors",
            "model.safetensors.index.json",
            "params.json",
            "config.json",
            "tekken.json",
        ],
    )
    return Path(path)


_ENC = r"whisper_encoder\.transformer\.layers\.(\d+)"
_LM = r"layers\.(\d+)"
_REMAP_PATTERNS = [
    (r"whisper_encoder\.conv_layers\.0\.conv\.(.*)",
     r"encoder.conv1.\1"),
    (r"whisper_encoder\.conv_layers\.1\.conv\.(.*)",
     r"encoder.conv2.\1"),
    (rf"{_ENC}\.attention\.wq\.(.*)",
     r"encoder.layers.\1.attention.q_proj.\2"),
    (rf"{_ENC}\.attention\.wk\.(.*)",
     r"encoder.layers.\1.attention.k_proj.\2"),
    (rf"{_ENC}\.attention\.wv\.(.*)",
     r"encoder.layers.\1.attention.v_proj.\2"),
    (rf"{_ENC}\.attention\.wo\.(.*)",
     r"encoder.layers.\1.attention.o_proj.\2"),
    (rf"{_ENC}\.attention_norm\.(.*)",
     r"encoder.layers.\1.attn_norm.\2"),
    (rf"{_ENC}\.feed_forward\.w1\.(.*)",
     r"encoder.layers.\1.mlp.gate_proj.\2"),
    (rf"{_ENC}\.feed_forward\.w2\.(.*)",
     r"encoder.layers.\1.mlp.down_proj.\2"),
    (rf"{_ENC}\.feed_forward\.w3\.(.*)",
     r"encoder.layers.\1.mlp.up_proj.\2"),
    (rf"{_ENC}\.ffn_norm\.(.*)",
     r"encoder.layers.\1.ffn_norm.\2"),
    (r"whisper_encoder\.transformer\.norm\.(.*)",
     r"encoder.norm.\1"),
    (r"audio_language_projection\.0\.weight",
     r"adapter.w_in.weight"),
    (r"audio_language_projection\.2\.weight",
     r"adapter.w_out.weight"),
    (r"tok_embeddings\.weight",
     r"language_model.embed_tokens.weight"),
    (rf"{_LM}\.attention\.wq\.weight",
     r"language_model.layers.\1.attention.q_proj.weight"),
    (rf"{_LM}\.attention\.wk\.weight",
     r"language_model.layers.\1.attention.k_proj.weight"),
    (rf"{_LM}\.attention\.wv\.weight",
     r"language_model.layers.\1.attention.v_proj.weight"),
    (rf"{_LM}\.attention\.wo\.weight",
     r"language_model.layers.\1.attention.o_proj.weight"),
    (rf"{_LM}\.attention_norm\.weight",
     r"language_model.layers.\1.attn_norm.weight"),
    (rf"{_LM}\.feed_forward\.w1\.weight",
     r"language_model.layers.\1.mlp.gate_proj.weight"),
    (rf"{_LM}\.feed_forward\.w2\.weight",
     r"language_model.layers.\1.mlp.down_proj.weight"),
    (rf"{_LM}\.feed_forward\.w3\.weight",
     r"language_model.layers.\1.mlp.up_proj.weight"),
    (rf"{_LM}\.ffn_norm\.weight",
     r"language_model.layers.\1.ffn_norm.weight"),
    (rf"{_LM}\.ada_rms_norm_t_cond\.0\.weight",
     r"language_model.layers.\1.ada_norm.linear_in.weight"),
    (rf"{_LM}\.ada_rms_norm_t_cond\.2\.weight",
     r"language_model.layers.\1.ada_norm.linear_out.weight"),
    (r"norm\.weight",
     r"language_model.norm.weight"),
]


def _remap_name(name: str) -> str | None:
    name = re.sub(
        r"^(mm_streams_embeddings\.embedding_module"
        r"|mm_whisper_embeddings)\.", "", name,
    )
    for pattern, replacement in _REMAP_PATTERNS:
        new_name, n = re.subn(f"^{pattern}$", replacement, name)
        if n > 0:
            return new_name
    return None


def _is_conv_weight(name: str) -> bool:
    return ("conv1.weight" in name or "conv2.weight" in name) and "bias" not in name


def _is_converted_format(model_path: Path) -> bool:
    has_config = (model_path / "config.json").exists()
    has_consolidated = (model_path / "consolidated.safetensors").exists()
    return has_config and not has_consolidated


def _load_converted(model_path: Path) -> tuple[VoxtralRealtime, dict]:
    with open(model_path / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    quant_config = config.get("quantization")
    model = VoxtralRealtime(config)

    if quant_config is not None:
        group_size = quant_config["group_size"]

        def predicate(_path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if module.weight.shape[-1] % group_size != 0:
                return False
            return True

        nn.quantize(
            model,
            group_size=group_size,
            bits=quant_config["bits"],
            class_predicate=predicate,
        )

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        weights = {}
        for shard_file in shard_files:
            weights.update(mx.load(str(model_path / shard_file)))
    else:
        weights = mx.load(str(model_path / "model.safetensors"))

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    return model, config


def _load_original(model_path: Path) -> tuple[VoxtralRealtime, dict]:
    with open(model_path / "params.json", encoding="utf-8") as f:
        config = json.load(f)

    model = VoxtralRealtime(config)
    weights = mx.load(str(model_path / "consolidated.safetensors"))

    remapped = {}
    skipped = []
    for name, tensor in weights.items():
        if name == "output.weight":
            continue
        new_name = _remap_name(name)
        if new_name is None:
            skipped.append(name)
            continue
        if _is_conv_weight(new_name):
            tensor = mx.swapaxes(tensor, 1, 2)
        remapped[new_name] = tensor

    if skipped:
        print(f"Warning: skipped {len(skipped)} unrecognized weights: {skipped[:5]}...")

    model.load_weights(list(remapped.items()))
    mx.eval(model.parameters())

    return model, config


def load_model(model_path: str | Path) -> tuple[VoxtralRealtime, dict]:
    model_path = Path(model_path)
    if not model_path.exists():
        model_path = download_model(str(model_path))
    if _is_converted_format(model_path):
        return _load_converted(model_path)
    return _load_original(model_path)


def load_tokenizer(model_path: str | Path):
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    model_path = Path(model_path)
    if not model_path.exists():
        model_path = download_model(str(model_path))
    tekken_path = model_path / "tekken.json"
    return Tekkenizer.from_file(str(tekken_path))
