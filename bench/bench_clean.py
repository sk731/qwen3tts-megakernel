"""End-to-end RTF under sampling (megakernel talker + code predictor)."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from decoders.megakernel import MegakernelTalker

TEXT = "The quick brown fox jumps over the lazy dog."


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def run(mk, label):
    torch.manual_seed(0)
    mk.model.generate_custom_voice(text=TEXT, speaker="Ryan", language="English",
                                   do_sample=True, subtalker_dosample=True)  # warmup
    torch.manual_seed(0)
    t = now()
    wavs, sr = mk.model.generate_custom_voice(text=TEXT, speaker="Ryan", language="English",
                                              do_sample=True, subtalker_dosample=True)
    dt = now() - t
    audio = len(wavs[0]) / sr
    print(f"{label:34s}: {dt*1000:7.0f} ms  audio {audio:.2f}s  RTF {dt/audio:.3f}", flush=True)


mk = MegakernelTalker(accel_codepred=True)
run(mk, "accel cp + isin-cached, no grab")
