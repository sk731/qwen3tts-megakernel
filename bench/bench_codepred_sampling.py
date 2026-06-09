"""Code predictor on the megakernel vs PyTorch, under sampling: RTF + waveform check."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import torch

from decoders.megakernel import MegakernelTalker

TEXT = "The quick brown fox jumps over the lazy dog."


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def run(mk, label, out_path):
    torch.manual_seed(0)
    mk.synthesize(TEXT, greedy=False)  # warmup
    torch.manual_seed(0)
    t = now()
    wav, sr, _ = mk.synthesize(TEXT, greedy=False)
    dt = now() - t
    audio = len(wav) / sr
    sf.write(out_path, np.asarray(wav, dtype=np.float32), sr)
    print(f"{label:30s}: {dt*1000:7.0f} ms  audio {audio:5.2f}s  RTF {dt/audio:.3f}  -> {out_path}",
          flush=True)
    return audio


mk = MegakernelTalker(accel_codepred=False)

mk._cp_accel = False
a_off = run(mk, "sampling, codepred PyTorch", "/workspace/out_sampled_pytorch.wav")

mk._cp_accel = True
a_on = run(mk, "sampling, codepred KERNEL", "/workspace/out_sampled_kernel.wav")

print(f"\nrunaway check: pytorch {a_off:.2f}s vs kernel {a_on:.2f}s "
      f"(similar => no degeneration)", flush=True)
