"""Chunked codec decode -> streaming PCM (re-decode growing prefix, emit new tail)."""

import numpy as np


def _to_numpy(w):
    return w.detach().cpu().numpy().reshape(-1) if hasattr(w, "detach") else np.asarray(w).reshape(-1)


def stream_pcm(speech_tokenizer, codes, chunk_frames=4):
    """Yield (pcm float32 @ 24kHz, sample_rate) as codec frames accumulate.

    codes: [num_frames, 16] int64. chunk_frames: frames per emit (4 = 320ms, the
    official packet; smaller = lower TTFC, more decode calls).
    """
    n = codes.shape[0]
    emitted = 0
    for end in range(chunk_frames, n + chunk_frames, chunk_frames):
        wavs, sr = speech_tokenizer.decode([{"audio_codes": codes[:min(end, n)]}])
        pcm = _to_numpy(wavs[0])
        yield pcm[emitted:], sr
        emitted = len(pcm)
