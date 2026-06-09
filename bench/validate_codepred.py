"""Validate the megakernel reproduces the code-predictor backbone (cosine vs stock PyTorch)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from decoders.megakernel import MegakernelTalker

S = 18  # one frame's worth: 2 prefill + up to 15 codebooks

mk = MegakernelTalker(accel_codepred=False)
bb = mk.model.model.talker.code_predictor.model

torch.manual_seed(0)
ie = torch.randn(1, S, 1024, device="cuda", dtype=torch.bfloat16)

# reference: stock backbone, fresh cache, standard causal positions 0..S-1
ref = mk._cp_orig_forward(
    inputs_embeds=ie, use_cache=True, past_key_values=DynamicCache(),
    cache_position=torch.arange(S, device="cuda"),
).last_hidden_state[0]  # [S, 1024]

# kernel: reset this backbone's KV, step position by position
mk._cp_k_cache.zero_()
mk._cp_v_cache.zero_()
mine = torch.stack([mk._cp_step(ie[0, t].contiguous(), t) for t in range(S)])  # [S, 1024]

cos = F.cosine_similarity(ref.float(), mine.float(), dim=-1)
print("per-position cosine (kernel vs stock 5-layer backbone):", flush=True)
for t in range(S):
    print(f"  pos {t:2d}: {cos[t].item():.4f}", flush=True)
print(f"min cosine {cos.min().item():.4f}  mean {cos.mean().item():.4f}", flush=True)
