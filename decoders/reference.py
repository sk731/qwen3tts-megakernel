"""Reference talker backend: stock qwen-tts in eager PyTorch. Runs anywhere (CPU default)."""

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from decoders.base import TalkerDecoder
from tts.mimi_decode import stream_pcm

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


class ReferenceTalker(TalkerDecoder):
    def __init__(self, model_id=MODEL, device="cpu", dtype=None):
        dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
        kwargs = dict(dtype=dtype, attn_implementation="eager")
        if device != "cpu":
            kwargs["device_map"] = device
        self.model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)

    def synthesize(self, text, speaker="Ryan", language="English", greedy=True):
        # talker's argmax token per step, grabbed off codec_head (one per ~80ms frame)
        toks = []

        def grab(_m, _i, out):
            out = out if isinstance(out, torch.Tensor) else out[0]
            toks.append(int(out.reshape(-1, out.shape[-1])[-1].argmax()))

        h = self.model.model.talker.codec_head.register_forward_hook(grab)
        try:
            wavs, sr = self.model.generate_custom_voice(
                text=text, speaker=speaker, language=language,
                do_sample=not greedy, subtalker_dosample=not greedy,
            )
        finally:
            h.remove()
        return wavs[0], sr, np.array(toks, dtype=np.int64)

    def stream(self, text, speaker="Ryan", language="English", chunk_frames=4):
        # qwen-tts can't stream tokens, so the talker decode runs fully here; we then
        # stream the OUTPUT via chunked codec decode. the megakernel makes token
        # production incremental too - same chunk-and-yield seam.
        codes = self._codes(text, speaker, language)
        yield from stream_pcm(self.model.model.speech_tokenizer, codes, chunk_frames)

    def _codes(self, text, speaker, language):
        """Full [num_frames, 16] codes, grabbed off the codec decode call."""
        st = self.model.model.speech_tokenizer
        holder = {}
        orig = st.decode

        def grab(arg, *a, **k):
            holder["codes"] = arg[0]["audio_codes"].detach().clone()
            return orig(arg, *a, **k)

        st.decode = grab
        try:
            self.model.generate_custom_voice(
                text=text, speaker=speaker, language=language,
                do_sample=False, subtalker_dosample=False,
            )
        finally:
            st.decode = orig
        return holder["codes"]
