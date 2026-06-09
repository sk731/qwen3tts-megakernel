"""Clean TTFC + bottleneck breakdown in ONE measure. Warm with non-streaming synthesize (no worker
thread), then measure exactly one streamed first chunk with per-phase timing. No O(n^2) drain, no
worker race. The first chunk is inherently serial, so the per-phase syncs don't inflate the total."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from decoders.megakernel import MegakernelTalker


def synct():
    torch.cuda.synchronize()
    return time.perf_counter()


mk = MegakernelTalker(accel_codepred=True)

# warm the generate + kernel + codec path with non-streaming synthesize (no worker thread,
# terminates cleanly). Two calls so any first-call lazy init is paid before we measure.
mk.synthesize("Hello there, how are you doing today?", "Ryan", greedy=False)
mk.synthesize("The quick brown fox jumps over the lazy dog.", "Ryan", greedy=False)

# instrument the three phases of the first chunk
talker_bb = mk.model.model.talker.model
cp = mk.model.model.talker.code_predictor
stk = mk.model.model.speech_tokenizer
rec = {"prefill": None, "talk": [], "cp": [], "codec": []}
o_fwd, o_gen, o_dec = talker_bb.forward, cp.generate, stk.decode


def w_fwd(*a, **k):
    ie = k.get("inputs_embeds", a[0] if a else None)
    seq = ie.shape[1] if ie is not None else 1
    t = synct(); out = o_fwd(*a, **k); dt = synct() - t
    rec.__setitem__("prefill", (seq, dt)) if seq > 1 else rec["talk"].append(dt)
    return out


def w_gen(*a, **k):
    t = synct(); out = o_gen(*a, **k); rec["cp"].append(synct() - t); return out


def w_dec(*a, **k):
    t = synct(); out = o_dec(*a, **k); rec["codec"].append(synct() - t); return out


talker_bb.forward, cp.generate, stk.decode = w_fwd, w_gen, w_dec

t0 = synct()
g = mk.stream("The quick brown fox jumps over the lazy dog.", chunk_frames=1)
first, sr = next(g)
ttfc = (synct() - t0) * 1000

pf = rec["prefill"]
print(f"CLEAN_TTFC={ttfc:.1f} ms (first chunk {len(first) / sr * 1000:.0f} ms audio)", flush=True)
if pf:
    print(f"  prefill         : {pf[0]} tokens  {pf[1] * 1000:.1f} ms  ({pf[1] / pf[0] * 1000:.2f} ms/tok)", flush=True)
print(f"  1st talker frame: {(rec['talk'][0] * 1000 if rec['talk'] else 0):.1f} ms", flush=True)
print(f"  1st cp (15 cb)  : {(rec['cp'][0] * 1000 if rec['cp'] else 0):.1f} ms", flush=True)
print(f"  1st codec decode: {(rec['codec'][0] * 1000 if rec['codec'] else 0):.1f} ms", flush=True)

os._exit(0)  # skip the messy CUDA/daemon-thread teardown
