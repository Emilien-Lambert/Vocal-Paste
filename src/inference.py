import threading
import queue
import time
import numpy as np
import src.config as config
from src.utils import log

N_LEFT_PAD_TOKENS = 32
N_RIGHT_PAD_TOKENS = 17
N_DELAY_TOKENS = 6
PREFIX_LEN = 1 + N_LEFT_PAD_TOKENS + N_DELAY_TOKENS  # BOS + 38 STREAMING_PAD


class InferenceService:
    def __init__(self):
        self.model = None
        self.sp = None
        self.t_cond = None
        self.text_embeds = None
        self.n_layers = 0
        self.eos_token_id = 0
        self.sliding_window = 8192
        self._audio_queue = queue.Queue()
        self._thread = None
        self._result = None
        self._stopped = False

    def initialize(self):
        log("Loading model...", verbose_only=True)
        import mlx.core as mx
        from src.weights import load_model, load_tokenizer

        self.model, _ = load_model(config.MODELS_DIR)
        self.sp = load_tokenizer(config.MODELS_DIR)

        streaming_pad = self.sp.get_special_token("[STREAMING_PAD]")
        prompt_tokens = [self.sp.bos_id] + [streaming_pad] * (N_LEFT_PAD_TOKENS + N_DELAY_TOKENS)
        self.eos_token_id = self.sp.eos_id

        self.t_cond = self.model.time_embedding(mx.array([N_DELAY_TOKENS], dtype=mx.float32))
        prompt_ids = mx.array([prompt_tokens])
        self.text_embeds = self.model.language_model.embed(prompt_ids)[0]
        mx.eval(self.t_cond, self.text_embeds)

        self.n_layers = len(self.model.language_model.layers)
        log("Model loaded.", verbose_only=True)

    def start_streaming(self):
        # Clear any leftover items
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._result = None
        self._stopped = False
        self._thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self._thread.start()

    def send_chunk(self, audio_chunk: np.ndarray):
        self._audio_queue.put(("CHUNK", audio_chunk))

    def stop_streaming(self) -> str:
        start_time = time.time()
        self._stopped = True
        self._audio_queue.put(("STOP",))
        self._thread.join(timeout=60)
        elapsed = time.time() - start_time
        log(f"Inference time: {elapsed:.2f}s", verbose_only=True)
        return self._result or ""

    def shutdown(self):
        pass

    def _streaming_worker(self):
        try:
            import mlx.core as mx
            from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
            from src.model import RotatingKVCache
            from src.mel import log_mel_spectrogram_step, SAMPLES_PER_TOKEN

            model = self.model
            sp = self.sp
            t_cond = self.t_cond
            text_embeds = self.text_embeds

            def sample(logits):
                return mx.argmax(logits[0, -1:], axis=-1).squeeze()

            # Encoder state
            audio_tail = None
            conv1_tail = None
            conv2_tail = None
            encoder_cache = None
            ds_buf = None

            # Decoder state
            decoder_cache = None
            y = None
            prefilled = False

            # Buffers
            pending_audio = np.zeros(0, dtype=np.float32)
            audio_embeds = None
            n_audio_samples_fed = 0
            n_total_decoded = 0
            first_cycle = True
            output_tokens = []

            def encode_chunk(chunk):
                nonlocal audio_tail, conv1_tail, conv2_tail, encoder_cache, ds_buf
                nonlocal audio_embeds

                mel, audio_tail = log_mel_spectrogram_step(chunk, audio_tail)
                new_embeds, conv1_tail, conv2_tail, encoder_cache, ds_buf = (
                    model.encode_step(mel, conv1_tail, conv2_tail, encoder_cache, ds_buf)
                )
                if new_embeds is not None:
                    mx.eval(new_embeds)
                    if audio_embeds is not None:
                        audio_embeds = mx.concatenate([audio_embeds, new_embeds])
                    else:
                        audio_embeds = new_embeds
                mx.clear_cache()

            def decode_available():
                nonlocal decoder_cache, y, prefilled, audio_embeds
                nonlocal n_total_decoded, output_tokens

                if audio_embeds is None:
                    return

                safe_total = N_LEFT_PAD_TOKENS + n_audio_samples_fed // SAMPLES_PER_TOKEN
                n_decodable = min(audio_embeds.shape[0], safe_total - n_total_decoded)
                if n_decodable <= 0:
                    return

                if not prefilled:
                    if n_total_decoded + audio_embeds.shape[0] < PREFIX_LEN:
                        return
                    decoder_cache = [RotatingKVCache(self.sliding_window) for _ in range(self.n_layers)]
                    prefix_embeds = (text_embeds + audio_embeds[:PREFIX_LEN])[None, :, :]
                    logits = model.decode(prefix_embeds, t_cond, "causal", decoder_cache)
                    mx.eval(logits, *[x for c in decoder_cache for x in (c.keys, c.values)])
                    y = sample(logits)
                    mx.async_eval(y)
                    audio_embeds = audio_embeds[PREFIX_LEN:]
                    n_total_decoded = PREFIX_LEN
                    prefilled = True
                    n_decodable = min(audio_embeds.shape[0], safe_total - n_total_decoded)

                if n_decodable <= 0:
                    return

                n_consumed = 0
                for i in range(n_decodable):
                    token_embed = model.language_model.embed(y.reshape(1, 1))[0, 0]
                    step_embed = (audio_embeds[i] + token_embed)[None, None, :]
                    logits = model.decode(step_embed, t_cond, mask=None, cache=decoder_cache)
                    next_y = sample(logits)
                    mx.async_eval(next_y)

                    token_id = y.item()
                    if token_id == self.eos_token_id:
                        break
                    output_tokens.append(token_id)
                    if config.VERBOSE and not self._stopped:
                        text = sp.decode([token_id], special_token_policy=SpecialTokenPolicy.IGNORE)
                        print(text, end="", flush=True)

                    if i > 0 and i % 64 == 0:
                        mx.clear_cache()
                    y = next_y
                    n_consumed = i + 1

                n_total_decoded += n_consumed
                if audio_embeds.shape[0] > n_consumed:
                    audio_embeds = audio_embeds[n_consumed:]
                else:
                    audio_embeds = None

            # ── Main streaming loop ───────────────────────────────────────
            while True:
                try:
                    msg = self._audio_queue.get(timeout=0.02)
                except queue.Empty:
                    decode_available()
                    continue

                if not isinstance(msg, tuple):
                    continue

                if msg[0] == "CHUNK":
                    chunk_data = msg[1]
                    pending_audio = np.append(pending_audio, chunk_data)
                    n_audio_samples_fed += len(chunk_data)

                    if first_cycle and len(pending_audio) >= SAMPLES_PER_TOKEN:
                        left_pad = np.zeros(N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32)
                        n_feed = (len(pending_audio) // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
                        chunk = np.concatenate([left_pad, pending_audio[:n_feed]])
                        pending_audio = pending_audio[n_feed:]
                        encode_chunk(chunk)
                        first_cycle = False
                    elif not first_cycle and len(pending_audio) >= SAMPLES_PER_TOKEN:
                        n_feed = (len(pending_audio) // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
                        chunk = pending_audio[:n_feed]
                        pending_audio = pending_audio[n_feed:]
                        encode_chunk(chunk)

                    decode_available()

                elif msg[0] == "STOP":
                    # Flush remaining audio + right padding
                    right_pad = np.zeros(N_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32)
                    if first_cycle:
                        left_pad = np.zeros(N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32)
                        flush_chunk = np.concatenate([left_pad, pending_audio, right_pad])
                        first_cycle = False
                    else:
                        flush_chunk = np.concatenate([pending_audio, right_pad])
                    n_audio_samples_fed += len(pending_audio)
                    pending_audio = np.zeros(0, dtype=np.float32)
                    encode_chunk(flush_chunk)

                    # Final prefill if not done yet
                    if audio_embeds is not None and not prefilled and audio_embeds.shape[0] >= PREFIX_LEN:
                        decoder_cache = [RotatingKVCache(self.sliding_window) for _ in range(self.n_layers)]
                        prefix_embeds = (text_embeds + audio_embeds[:PREFIX_LEN])[None, :, :]
                        logits = model.decode(prefix_embeds, t_cond, "causal", decoder_cache)
                        mx.eval(logits, *[x for c in decoder_cache for x in (c.keys, c.values)])
                        y = sample(logits)
                        mx.async_eval(y)
                        audio_embeds = audio_embeds[PREFIX_LEN:]
                        n_total_decoded = PREFIX_LEN
                        prefilled = True

                    # Decode all remaining
                    if prefilled and audio_embeds is not None and y is not None:
                        for i in range(audio_embeds.shape[0]):
                            token_embed = model.language_model.embed(y.reshape(1, 1))[0, 0]
                            step_embed = (audio_embeds[i] + token_embed)[None, None, :]
                            logits = model.decode(step_embed, t_cond, mask=None, cache=decoder_cache)
                            next_y = sample(logits)
                            mx.async_eval(next_y)
                            token_id = y.item()
                            if token_id == self.eos_token_id:
                                break
                            output_tokens.append(token_id)
                            if i > 0 and i % 64 == 0:
                                mx.clear_cache()
                            y = next_y

                        # Flush last pending token
                        if y is not None:
                            token_id = y.item()
                            if token_id != self.eos_token_id:
                                output_tokens.append(token_id)
                    self._result = sp.decode(output_tokens, special_token_policy=SpecialTokenPolicy.IGNORE).strip()

                    # Free all session memory
                    del decoder_cache, encoder_cache, audio_embeds, ds_buf
                    del conv1_tail, conv2_tail, audio_tail, y
                    mx.clear_cache()
                    return  # end thread

        except Exception as e:
            log(f"ERROR: Streaming worker crash: {e}")
            self._result = f"ERROR: {e}"
        finally:
            mx.clear_cache()
