import math

import mlx.core as mx
import mlx.nn as nn


# ── KV Cache ──────────────────────────────────────────────────────────────────

class RotatingKVCache:
    step = 256

    def __init__(self, max_size):
        self.keys = None
        self.values = None
        self._offset = 0
        self.max_size = max_size
        self._idx = 0

    @property
    def offset(self):
        return self._offset

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., trim_size:, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self._offset:
            return mx.concatenate(
                [v[..., self._idx:, :], v[..., : self._idx, :]],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]

            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        if self._idx == self.max_size:
            self._idx = 0

        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self._idx += S

        if self._offset < self.max_size:
            return self.keys[..., : self._offset, :], self.values[..., : self._offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)


# ── Encoder ───────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding_total = kernel_size - stride
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding_total > 0:
            x = mx.pad(x, [(0, 0), (self.padding_total, 0), (0, 0)])
        return mx.conv1d(x, self.weight, stride=self.stride) + self.bias


class EncoderAttention(nn.Module):
    def __init__(self, dim: int = 1280, n_heads: int = 32, head_dim: int = 64, rope_theta: float = 1e6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=True)
        self.rope_theta = rope_theta

    def __call__(self, x: mx.array, offset: int, mask: mx.array, cache=None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
        q = mx.fast.rope(q, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class EncoderSwiGLU(nn.Module):
    def __init__(self, dim: int = 1280, hidden_dim: int = 5120):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int = 1280, n_heads: int = 32, head_dim: int = 64, hidden_dim: int = 5120, rope_theta: float = 1e6):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.attention = EncoderAttention(dim, n_heads, head_dim, rope_theta)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = EncoderSwiGLU(dim, hidden_dim)

    def __call__(self, x: mx.array, offset: int, mask: mx.array, cache=None) -> mx.array:
        x = x + self.attention(self.attn_norm(x), offset, mask, cache=cache)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class CausalWhisperEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        dim: int = 1280,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 64,
        hidden_dim: int = 5120,
        rope_theta: float = 1e6,
        sliding_window: int = 750,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, dim, kernel_size=3, stride=1)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=3, stride=2)
        self.layers = [
            EncoderLayer(dim, n_heads, head_dim, hidden_dim, rope_theta) for _ in range(n_layers)
        ]
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.sliding_window = sliding_window

    def forward_conv(self, mel: mx.array) -> mx.array:
        x = mel.T[None, :, :]
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        return x

    def forward_conv_step(self, new_mel, conv1_tail, conv2_tail):
        if conv1_tail is not None:
            x = mx.concatenate([conv1_tail, new_mel], axis=1)
        else:
            x = mx.pad(new_mel, [(0, 0), (self.conv1.padding_total, 0), (0, 0)])
        new_conv1_tail = new_mel[:, -self.conv1.padding_total:, :]
        x = nn.gelu(
            mx.conv1d(x, self.conv1.weight, stride=self.conv1.stride) + self.conv1.bias
        )

        if conv2_tail is not None:
            x_in = mx.concatenate([conv2_tail, x], axis=1)
        else:
            x_in = mx.pad(x, [(0, 0), (self.conv2.padding_total, 0), (0, 0)])
        new_conv2_tail = x[:, -self.conv2.padding_total:, :]
        x = nn.gelu(
            mx.conv1d(x_in, self.conv2.weight, stride=self.conv2.stride) + self.conv2.bias
        )

        return x, new_conv1_tail, new_conv2_tail

    def forward_transformer(self, x, cache=None):
        mask = "causal"
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, offset=0, mask=mask, cache=layer_cache)
        x = self.norm(x)
        return x

    def __call__(self, mel: mx.array) -> mx.array:
        x = self.forward_conv(mel.astype(self.conv1.weight.dtype))
        mask = "causal"
        for layer in self.layers:
            x = layer(x, offset=0, mask=mask)
        x = self.norm(x)
        return x


# ── Language Model (Decoder) ─────────────────────────────────────────────────

class DecoderAttention(nn.Module):
    def __init__(self, dim: int = 3072, n_heads: int = 32, n_kv_heads: int = 8, head_dim: int = 128, rope_theta: float = 1e6):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rope_theta = rope_theta

    def __call__(self, x: mx.array, mask=None, cache: RotatingKVCache | None = None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = mx.fast.rope(q, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=True, base=self.rope_theta, scale=1.0, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class DecoderSwiGLU(nn.Module):
    def __init__(self, dim: int = 3072, hidden_dim: int = 9216):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class AdaptiveNorm(nn.Module):
    def __init__(self, dim: int = 3072, cond_dim: int = 32):
        super().__init__()
        self.linear_in = nn.Linear(dim, cond_dim, bias=False)
        self.linear_out = nn.Linear(cond_dim, dim, bias=False)

    def __call__(self, t_cond: mx.array) -> mx.array:
        return self.linear_out(nn.gelu(self.linear_in(t_cond)))


class DecoderLayer(nn.Module):
    def __init__(self, dim: int = 3072, n_heads: int = 32, n_kv_heads: int = 8, head_dim: int = 128, hidden_dim: int = 9216, rope_theta: float = 1e6, cond_dim: int = 32):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.attention = DecoderAttention(dim, n_heads, n_kv_heads, head_dim, rope_theta)
        self.ada_norm = AdaptiveNorm(dim, cond_dim)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = DecoderSwiGLU(dim, hidden_dim)

    def __call__(self, x: mx.array, t_cond: mx.array, mask=None, cache: RotatingKVCache | None = None) -> mx.array:
        h = self.attention(self.attn_norm(x), mask, cache)
        x = x + h
        ffn_in = self.ffn_norm(x) * (1.0 + self.ada_norm(t_cond))
        x = x + self.mlp(ffn_in)
        return x


class LanguageModel(nn.Module):
    def __init__(self, dim: int = 3072, n_layers: int = 26, n_heads: int = 32, n_kv_heads: int = 8, head_dim: int = 128, hidden_dim: int = 9216, vocab_size: int = 131072, rope_theta: float = 1e6, cond_dim: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [
            DecoderLayer(dim, n_heads, n_kv_heads, head_dim, hidden_dim, rope_theta, cond_dim)
            for _ in range(n_layers)
        ]
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self._dim = dim

    def embed(self, input_ids: mx.array) -> mx.array:
        return self.embed_tokens(input_ids)

    def __call__(self, x: mx.array, t_cond: mx.array, mask=None, cache: list[RotatingKVCache] | None = None) -> mx.array:
        t_cond = t_cond.astype(x.dtype)
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, t_cond, mask, layer_cache)
        x = self.norm(x)
        logits = self.embed_tokens.as_linear(x)
        return logits


# ── Top-level Model ──────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = mx.exp(
            -math.log(theta) * mx.arange(dim // 2).astype(mx.float32) / (dim // 2)
        )
        self._inv_freq = inv_freq

    def __call__(self, t: mx.array) -> mx.array:
        t = t.reshape(-1, 1).astype(mx.float32)
        emb = t * self._inv_freq
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


class AudioLanguageAdapter(nn.Module):
    def __init__(self, in_dim: int = 5120, out_dim: int = 3072):
        super().__init__()
        self.w_in = nn.Linear(in_dim, out_dim, bias=False)
        self.w_out = nn.Linear(out_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_out(nn.gelu(self.w_in(x)))


class VoxtralRealtime(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        enc = config["multimodal"]["whisper_model_args"]["encoder_args"]
        audio_enc = enc["audio_encoding_args"]
        downsample = config["multimodal"]["whisper_model_args"]["downsample_args"]["downsample_factor"]

        self.encoder = CausalWhisperEncoder(
            in_channels=audio_enc["num_mel_bins"],
            dim=enc["dim"],
            n_layers=enc["n_layers"],
            n_heads=enc["n_heads"],
            head_dim=enc["head_dim"],
            hidden_dim=enc["hidden_dim"],
            rope_theta=enc["rope_theta"],
            sliding_window=enc["sliding_window"],
        )

        adapter_in = enc["dim"] * downsample
        self.adapter = AudioLanguageAdapter(adapter_in, config["dim"])

        cond_dim = config.get("ada_rms_norm_t_cond_dim", 32)
        self.language_model = LanguageModel(
            dim=config["dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            n_kv_heads=config["n_kv_heads"],
            head_dim=config["head_dim"],
            hidden_dim=config["hidden_dim"],
            vocab_size=config["vocab_size"],
            rope_theta=config["rope_theta"],
            cond_dim=cond_dim,
        )

        self.time_embedding = TimeEmbedding(dim=config["dim"])
        self.downsample_factor = downsample
        self._encoder_dim = enc["dim"]

    def encode(self, mel: mx.array) -> mx.array:
        T = mel.shape[1]
        if T % 2 != 0:
            mel = mel[:, 1:]

        x = self.encoder(mel)
        x = x[0]

        L = x.shape[0]
        remainder = L % self.downsample_factor
        if remainder != 0:
            x = x[remainder:]
            L = x.shape[0]

        x = x.reshape(L // self.downsample_factor, -1)
        x = self.adapter(x)
        return x

    def encode_step(self, new_mel, conv1_tail, conv2_tail, encoder_cache, ds_buf):
        x_mel = new_mel.T[None, :, :].astype(self.encoder.conv1.weight.dtype)

        x, conv1_tail, conv2_tail = self.encoder.forward_conv_step(
            x_mel, conv1_tail, conv2_tail
        )

        if encoder_cache is None:
            encoder_cache = [
                RotatingKVCache(self.encoder.sliding_window)
                for _ in range(len(self.encoder.layers))
            ]

        x = self.encoder.forward_transformer(x, cache=encoder_cache)
        x = x[0]

        if ds_buf is not None:
            x = mx.concatenate([ds_buf, x])
        n_complete = (x.shape[0] // self.downsample_factor) * self.downsample_factor
        if n_complete == 0:
            return None, conv1_tail, conv2_tail, encoder_cache, x

        ds_buf = x[n_complete:] if x.shape[0] > n_complete else None
        x = x[:n_complete]

        x = x.reshape(n_complete // self.downsample_factor, -1)
        x = self.adapter(x)
        return x, conv1_tail, conv2_tail, encoder_cache, ds_buf

    def decode(self, embeddings: mx.array, t_cond: mx.array, mask=None, cache: list | None = None):
        return self.language_model(embeddings, t_cond, mask, cache)
