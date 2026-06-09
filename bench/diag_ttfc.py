"""Diagnose where TTFC goes: clean TTFC, then a per-phase breakdown (prefill step count + ms,
first talker frame, code predictor, codec, and the generate() remainder). Run this FIRST on the
box - it tells us which lever to pull instead of guessing."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from decoders.megakernel import MegakernelTalker

TEXT = "The quick brown fox jumps over the lazy dog."


def synct():
    torch.cuda.synchronize()
    return time.perf_counter()


mk = MegakernelTalker(accel_codepred=True)

for _ in mk.stream("Hi there friend.", chunk_frames=1):  # warm the whole path
    pass

# 1) clean TTFC, no instrumentation (matches bench_ttfc)
t0 = synct()
g = mk.stream(TEXT, chunk_frames=1)
first, sr = next(g)
clean = synct() - t0
for _ in g:
    pass
print(f"clean TTFC: {clean * 1000:.1f} ms  (first {len(first) / sr * 1000:.0f} ms of audio)", flush=True)

# 2) instrumented breakdown (syncs add a little, but the split is what we want)
talker_bb = mk.model.model.talker.model
cp = mk.model.model.talker.code_predictor
stk = mk.model.model.speech_tokenizer
rec = {"prefill": None, "talk": [], "cp": [], "codec": []}

o_fwd = talker_bb.forward
def w_fwd(*a, **k):
    ie = k.get("inputs_embeds", a[0] if a else None)
    seq = ie.shape[1] if ie is not None else 1
    t = synct(); out = o_fwd(*a, **k); dt = synct() - t
    rec.__setitem__("prefill", (seq, dt)) if seq > 1 else rec["talk"].append(dt)
    return out
talker_bb.forward = w_fwd

o_gen = cp.generate
def w_gen(*a, **k):
    t = synct(); out = o_gen(*a, **k); rec["cp"].append(synct() - t); return out
cp.generate = w_gen

o_dec = stk.decode
def w_dec(*a, **k):
    t = synct(); out = o_dec(*a, **k); rec["codec"].append(synct() - t); return out
stk.decode = w_dec

t0 = synct()
g = mk.stream(TEXT, chunk_frames=1)
first, sr = next(g)
inst = synct() - t0
for _ in g:
    pass

pf = rec["prefill"]
talk0 = rec["talk"][0] if rec["talk"] else 0.0
cp0 = rec["cp"][0] if rec["cp"] else 0.0
cod0 = rec["codec"][0] if rec["codec"] else 0.0
known = (pf[1] if pf else 0) + talk0 + cp0 + cod0

print("\n--- breakdown to first chunk (syncs inflate the total a touch) ---", flush=True)
if pf:
    print(f"prefill          : {pf[0]:3d} prompt tokens  {pf[1] * 1000:7.1f} ms  "
          f"({pf[1] / pf[0] * 1000:.2f} ms/token)  <-- is the prompt long?", flush=True)
print(f"1st talker frame : {talk0 * 1000:7.1f} ms", flush=True)
print(f"1st cp (15 cb)   : {cp0 * 1000:7.1f} ms", flush=True)
print(f"1st codec decode : {cod0 * 1000:7.1f} ms", flush=True)
print(f"remainder        : {(inst - known) * 1000:7.1f} ms  <-- tokenize + prompt-build + "
      f"HF generate() overhead", flush=True)
print(f"instrumented TTFC: {inst * 1000:.1f} ms", flush=True)
