"""Megakernel decode backend for the Qwen3-TTS talker and code predictor. RTX 5090 only."""

import math
import os
import struct
import sys
from types import SimpleNamespace

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from decoders.base import TalkerDecoder

MODEL = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, HIDDEN, INTER = 28, 8, 128, 1024, 3072
Q_SIZE, KV_SIZE, MAX_SEQ, VOCAB = 16 * HEAD_DIM, 8 * HEAD_DIM, 2048, 151936
ROPE_THETA = 1000000.0  # talker's rope_theta (verified)

# Code predictor (sub-talker) backbone: identical per-head shape to the talker, only
# 5 layers and a short context (it restarts per frame, max ~16 codebook steps). Same
# kernel drives it with NUM_LAYERS_CP. cos/sin + scratch buffers are shared with the
# talker (single CUDA stream, never concurrent); only weights and KV cache are its own.
NUM_LAYERS_CP, CP_MAX_SEQ = 5, 64


def _patch_suppress_tokens():
    # transformers' SuppressTokensLogitsProcessor rebuilds a CONSTANT isin() mask over the
    # whole vocab on EVERY decode step - ~25% of the talker's generation time (cProfile).
    # The mask only depends on (vocab_size, suppress_tokens), both fixed, so cache it. Same
    # output, computed once.
    from transformers.generation.logits_process import SuppressTokensLogitsProcessor
    from transformers.pytorch_utils import isin_mps_friendly

    if getattr(SuppressTokensLogitsProcessor, "_mask_cached", False):
        return

    def cached_call(self, input_ids, scores):
        m = getattr(self, "_cm", None)
        if m is None or m.shape[-1] != scores.shape[-1] or m.device != scores.device:
            vocab = torch.arange(scores.shape[-1], device=scores.device)
            self._cm = isin_mps_friendly(vocab, self.suppress_tokens.to(scores.device))
        return torch.where(self._cm, -float("inf"), scores)

    SuppressTokensLogitsProcessor.__call__ = cached_call
    SuppressTokensLogitsProcessor._mask_cached = True


def _build_ops(kernel_path=None):
    # the backend imports the kernel from here and JIT-builds it. clone qwen_megakernel to the
    # default path below, or point QWEN_MEGAKERNEL_PATH at wherever you cloned it.
    kernel_path = kernel_path or os.getenv("QWEN_MEGAKERNEL_PATH", "/workspace/qwen_megakernel")
    if kernel_path not in sys.path:
        sys.path.insert(0, kernel_path)
    from qwen_megakernel.build import get_extension
    get_extension()  # JIT build / load torch.ops.qwen_megakernel_C
    return torch.ops.qwen_megakernel_C.decode


class MegakernelTalker(TalkerDecoder):
    def __init__(self, model_id=MODEL, accel_codepred=False):
        _patch_suppress_tokens()
        self._decode = _build_ops()
        self.model = Qwen3TTSModel.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
        )
        self._pos = 0
        self._setup_kernel()
        self._patch_backbone()
        self._setup_codepred()
        self._patch_codepred()
        self._patch_codepred_generate()
        self._cp_accel = accel_codepred
        if os.getenv("SKIP_CP_GRAPH"):
            self._cp_nogr = True  # CUDA-graph capture corrupts state on some 5090 instances;
            # fall back to the no-graph kernel rollout (still on the megakernel, just per-launch)
        else:
            self._warmup_cp_graph()
        self._warmup_pipeline()

    def _warmup_pipeline(self):
        # The streaming first chunk decodes ONE frame, whose conv shape cudnn benchmarks cold
        # (~100 ms - the single biggest TTFC cost). Warm it here with a direct 1-frame decode:
        # no generation, so no runaway risk, so the user's first reply isn't cudnn-cold.
        ng = self.model.model.talker.config.num_code_groups
        try:
            self.model.model.speech_tokenizer.decode(
                [{"audio_codes": torch.zeros(1, ng, dtype=torch.long, device="cuda")}])
        except Exception:
            pass

    def _setup_kernel(self):
        sd = self.model.model.talker.model.state_dict()

        def g(k):
            return sd[k].to(torch.bfloat16).cuda().contiguous()

        lw = []
        for i in range(NUM_LAYERS):
            p = f"layers.{i}."
            lw += [g(p + "input_layernorm.weight"), g(p + "self_attn.q_proj.weight"),
                   g(p + "self_attn.k_proj.weight"), g(p + "self_attn.v_proj.weight"),
                   g(p + "self_attn.q_norm.weight"), g(p + "self_attn.k_norm.weight"),
                   g(p + "self_attn.o_proj.weight"), g(p + "post_attention_layernorm.weight"),
                   g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight"),
                   g(p + "mlp.down_proj.weight")]
        self._lw = lw  # keep refs alive
        self._final_norm = g("norm.weight")
        buf = bytearray(NUM_LAYERS * 11 * 8)
        for i in range(NUM_LAYERS * 11):
            struct.pack_into("Q", buf, i * 8, lw[i].data_ptr())
        self._packed = torch.frombuffer(bytearray(buf), dtype=torch.uint8).cuda()

        inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        fr = torch.outer(torch.arange(MAX_SEQ, dtype=torch.float32), inv)
        self._cos = torch.cos(fr).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
        self._sin = torch.sin(fr).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

        bf = dict(dtype=torch.bfloat16, device="cuda")
        f = dict(dtype=torch.float32, device="cuda")
        self._k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, **bf)
        self._v_cache = torch.zeros_like(self._k_cache)
        self._hidden = torch.empty(HIDDEN, **bf)
        self._act = torch.empty(HIDDEN, **f)
        self._res = torch.empty(HIDDEN, **f)
        self._q = torch.empty(Q_SIZE, **f)
        self._k = torch.empty(KV_SIZE, **f)
        self._v = torch.empty(KV_SIZE, **f)
        self._attn = torch.empty(Q_SIZE, **f)
        self._mlp = torch.empty(INTER, **f)
        self._norm = torch.empty(HIDDEN, **f)
        self._bmv = torch.empty(4096, **f)
        self._bmi = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_tok = torch.empty(1, dtype=torch.int32, device="cuda")
        self._dummy_lm = torch.zeros(VOCAB, HIDDEN, **bf)
        self._scale = 1.0 / math.sqrt(HEAD_DIM)

    def _step(self, input_embed, position):
        self._decode(self._out_tok, 0, input_embed, self._packed, self._final_norm,
                     self._dummy_lm, self._cos, self._sin, self._k_cache, self._v_cache,
                     self._hidden, self._act, self._res, self._q, self._k, self._v,
                     self._attn, self._mlp, self._norm, self._bmv, self._bmi,
                     NUM_LAYERS, position, MAX_SEQ, self._scale)
        return self._norm.clone()

    def _patch_backbone(self):
        bb = self.model.model.talker.model

        def forward(inputs_embeds=None, cache_position=None, past_key_values=None, **kw):
            seq = inputs_embeds.shape[1]
            if seq > 1:  # prefill = new utterance: reset kernel KV cache + position
                self._k_cache.zero_()
                self._v_cache.zero_()
                self._pos = 0
            ie = inputs_embeds.to(torch.bfloat16)
            hs = []
            for t in range(seq):
                hs.append(self._step(ie[0, t].contiguous(), self._pos))
                self._pos += 1
            last = torch.stack(hs).unsqueeze(0).to(inputs_embeds.dtype)  # [1, seq, 1024]
            return BaseModelOutputWithPast(
                last_hidden_state=last, past_key_values=past_key_values,
                hidden_states=(last,), attentions=None)

        bb.forward = forward

    def _setup_codepred(self):
        sd = self.model.model.talker.code_predictor.model.state_dict()

        def g(k):
            return sd[k].to(torch.bfloat16).cuda().contiguous()

        lw = []
        for i in range(NUM_LAYERS_CP):
            p = f"layers.{i}."
            lw += [g(p + "input_layernorm.weight"), g(p + "self_attn.q_proj.weight"),
                   g(p + "self_attn.k_proj.weight"), g(p + "self_attn.v_proj.weight"),
                   g(p + "self_attn.q_norm.weight"), g(p + "self_attn.k_norm.weight"),
                   g(p + "self_attn.o_proj.weight"), g(p + "post_attention_layernorm.weight"),
                   g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight"),
                   g(p + "mlp.down_proj.weight")]
        self._cp_lw = lw  # keep refs alive
        self._cp_final_norm = g("norm.weight")
        buf = bytearray(NUM_LAYERS_CP * 11 * 8)
        for i in range(NUM_LAYERS_CP * 11):
            struct.pack_into("Q", buf, i * 8, lw[i].data_ptr())
        self._cp_packed = torch.frombuffer(bytearray(buf), dtype=torch.uint8).cuda()

        bf = dict(dtype=torch.bfloat16, device="cuda")
        self._cp_k_cache = torch.zeros(NUM_LAYERS_CP, NUM_KV_HEADS, CP_MAX_SEQ, HEAD_DIM, **bf)
        self._cp_v_cache = torch.zeros_like(self._cp_k_cache)
        self._cp_pos = 0

    def _cp_step(self, input_embed, position):
        self._decode(self._out_tok, 0, input_embed, self._cp_packed, self._cp_final_norm,
                     self._dummy_lm, self._cos, self._sin, self._cp_k_cache, self._cp_v_cache,
                     self._hidden, self._act, self._res, self._q, self._k, self._v,
                     self._attn, self._mlp, self._norm, self._bmv, self._bmi,
                     NUM_LAYERS_CP, position, CP_MAX_SEQ, self._scale)
        return self._norm.clone()

    def _patch_codepred(self):
        bb = self.model.model.talker.code_predictor.model
        self._cp_orig_forward = bb.forward

        def forward(inputs_embeds=None, cache_position=None, past_key_values=None, **kw):
            if not self._cp_accel:  # A/B: fall back to stock PyTorch backbone
                return self._cp_orig_forward(
                    inputs_embeds=inputs_embeds, cache_position=cache_position,
                    past_key_values=past_key_values, **kw)
            # Track position ourselves like the talker patch: cache_position is derived
            # from the real DynamicCache, which we bypass, so it's stuck at 0. Each frame
            # opens with the 2-token prefill [past_hidden, codebook0] (seq > 1) -> reset.
            seq = inputs_embeds.shape[1]
            if seq > 1:
                self._cp_k_cache.zero_()
                self._cp_v_cache.zero_()
                self._cp_pos = 0
            ie = inputs_embeds.to(torch.bfloat16)
            hs = []
            for t in range(seq):
                hs.append(self._cp_step(ie[0, t].contiguous(), self._cp_pos))
                self._cp_pos += 1
            last = torch.stack(hs).unsqueeze(0).to(inputs_embeds.dtype)
            return BaseModelOutputWithPast(
                last_hidden_state=last, past_key_values=past_key_values,
                hidden_states=(last,), attentions=None)

        bb.forward = forward

    def _patch_codepred_generate(self):
        # The talker calls code_predictor.generate() once per frame to roll out the 15
        # extra codebooks. transformers' generate() spends ~5 ms/token on host machinery
        # (input prep, logits warpers, stopping criteria). We drive the same rollout by
        # hand: one kernel step per codebook, a tiny PyTorch lm_head + sample, feed back.
        # Then capture the whole rollout in a CUDA graph so a frame is one replay, not
        # ~90 Python-dispatched launches. The custom decode op is capture-safe (no per-call
        # malloc/host-sync after warmup), proven in bench/capture_test.py.
        cp = self.model.model.talker.code_predictor
        self._cp_orig_generate = cp.generate
        self._cp_proj = cp.small_to_mtp_projection   # Identity when hidden sizes match
        self._cp_codec_emb = cp.model.codec_embedding
        self._cp_lm_head = cp.lm_head
        self._cp_graph = None
        self._cp_nogr = False  # set if graph capture fails -> stay on the eager loop
        self._cp_native = False  # drive the cp on stock PyTorch (5-layer) instead of the kernel
        n_groups = self.model.model.talker.config.num_code_groups

        def manual_generate(inputs_embeds=None, max_new_tokens=n_groups - 1, do_sample=True,
                             top_p=1.0, top_k=50, temperature=1.0, **kw):
            if not self._cp_accel:
                return self._cp_orig_generate(
                    inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
                    do_sample=do_sample, top_p=top_p, top_k=top_k,
                    temperature=temperature, **kw)
            samp = (do_sample, temperature, top_k, top_p, max_new_tokens)
            if self._cp_native:  # stock 5-layer backbone, no megakernel per-launch overhead
                return SimpleNamespace(sequences=self._cp_rollout_native(inputs_embeds, samp))
            ie = self._cp_proj(inputs_embeds).to(torch.bfloat16)  # [1, 2, H]
            if self._cp_graph is None and not self._cp_nogr:
                try:
                    self._build_cp_graph(samp)
                except Exception:
                    self._cp_nogr = True  # capture unsupported here -> eager fallback
            if self._cp_graph is not None and self._cp_samp == samp:
                self._cp_g_in.copy_(ie)
                self._cp_graph.replay()
                return SimpleNamespace(sequences=self._cp_g_out.clone())
            return SimpleNamespace(sequences=self._cp_rollout_eager(ie, samp))

        cp.generate = manual_generate

    def _cp_rollout(self, samp):
        # One frame: prefill [talker_hidden, codebook0] then 14 fed-back codebook steps.
        # Reads self._cp_g_in, writes sampled codes into self._cp_g_out (both static, so
        # this body is CUDA-graph-capturable).
        do_sample, temperature, top_k, top_p, n = samp
        ie = self._cp_g_in
        self._cp_k_cache.zero_()
        self._cp_v_cache.zero_()
        self._cp_step(ie[0, 0].contiguous(), 0)
        h = self._cp_step(ie[0, 1].contiguous(), 1)
        tok = self._cp_sample(self._cp_lm_head[0](h.to(torch.bfloat16)),
                              do_sample, temperature, top_k, top_p)
        self._cp_g_out[:, 0:1].copy_(tok)
        for gs in range(1, n):
            emb = self._cp_proj(self._cp_codec_emb[gs - 1](tok)).to(torch.bfloat16)
            h = self._cp_step(emb[0, 0].contiguous(), gs + 1)
            tok = self._cp_sample(self._cp_lm_head[gs](h.to(torch.bfloat16)),
                                  do_sample, temperature, top_k, top_p)
            self._cp_g_out[:, gs:gs + 1].copy_(tok)

    def _cp_rollout_eager(self, ie, samp):
        do_sample, temperature, top_k, top_p, n = samp
        self._cp_k_cache.zero_()
        self._cp_v_cache.zero_()
        self._cp_step(ie[0, 0].contiguous(), 0)
        h = self._cp_step(ie[0, 1].contiguous(), 1)
        toks = [self._cp_sample(self._cp_lm_head[0](h.to(torch.bfloat16)),
                                do_sample, temperature, top_k, top_p)]
        for gs in range(1, n):
            emb = self._cp_proj(self._cp_codec_emb[gs - 1](toks[-1])).to(torch.bfloat16)
            h = self._cp_step(emb[0, 0].contiguous(), gs + 1)
            toks.append(self._cp_sample(self._cp_lm_head[gs](h.to(torch.bfloat16)),
                                        do_sample, temperature, top_k, top_p))
        return torch.cat(toks, dim=-1)

    def _warmup_cp_graph(self):
        # Capture must happen in a clean stream context. Doing it lazily inside the talker's
        # generate loop fails (the talker has work in flight), so build it now, once, with
        # the model's default subtalker sampling (do_sample, temp 0.9, top_k 50, top_p 1.0).
        n = self.model.model.talker.config.num_code_groups - 1
        try:
            self._build_cp_graph((True, 0.9, 50, 1.0, n))
        except Exception:
            self._cp_nogr = True

    def _cp_rollout_native(self, inputs_embeds, samp):
        # Same rollout, but the per-codebook backbone step runs on the stock 5-layer PyTorch
        # model (sdpa) with a fresh DynamicCache - no megakernel per-launch fixed cost and no
        # dead 151936 lm_head. Correct by construction (it IS the reference backbone).
        do_sample, temperature, top_k, top_p, n = samp
        cache = DynamicCache()
        ie = self._cp_proj(inputs_embeds)
        out = self._cp_orig_forward(inputs_embeds=ie, past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[0, -1]
        toks = [self._cp_sample(self._cp_lm_head[0](h), do_sample, temperature, top_k, top_p)]
        for gs in range(1, n):
            emb = self._cp_proj(self._cp_codec_emb[gs - 1](toks[-1]))
            out = self._cp_orig_forward(inputs_embeds=emb, past_key_values=cache, use_cache=True)
            h = out.last_hidden_state[0, -1]
            toks.append(self._cp_sample(self._cp_lm_head[gs](h),
                                        do_sample, temperature, top_k, top_p))
        return torch.cat(toks, dim=-1)

    def _build_cp_graph(self, samp):
        n = samp[-1]
        self._cp_g_in = torch.zeros(1, 2, HIDDEN, dtype=torch.bfloat16, device="cuda")
        self._cp_g_out = torch.zeros(1, n, dtype=torch.long, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._cp_rollout(samp)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._cp_rollout(samp)
        self._cp_graph = graph
        self._cp_samp = samp

    def _cp_sample(self, logits, do_sample, temperature, top_k, top_p):
        logits = logits.reshape(-1).float()
        if not do_sample:
            return logits.argmax().view(1, 1)
        if temperature and temperature != 1.0:
            logits = logits / temperature
        if top_k and top_k < logits.numel():
            kth = torch.topk(logits, top_k).values[-1]
            logits = torch.where(logits < kth, logits.new_full((), float("-inf")), logits)
        if top_p and top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            remove = torch.softmax(sl, -1).cumsum(-1) > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sl[remove] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(0, si, sl)
        return torch.multinomial(torch.softmax(logits, -1), 1).view(1, 1)

    def synthesize(self, text, speaker="Ryan", language="English", greedy=True):
        toks = []

        def grab(_m, _i, out):
            out = out if isinstance(out, torch.Tensor) else out[0]
            toks.append(int(out.reshape(-1, out.shape[-1])[-1].argmax()))

        h = self.model.model.talker.codec_head.register_forward_hook(grab)
        try:
            wavs, sr = self.model.generate_custom_voice(
                text=text, speaker=speaker, language=language,
                do_sample=not greedy, subtalker_dosample=not greedy)
        finally:
            h.remove()
        return wavs[0], sr, np.array(toks, dtype=np.int64)

    def stream(self, text, speaker="Ryan", language="English", chunk_frames=1):
        """True frame-by-frame streaming: yield (pcm_float32, sr) as codes are generated.
        talker.forward returns each frame's 16 codes in hidden_states[1]; a forward hook
        queues them (a hook, not a forward swap, so generate's kwarg validation still sees
        the real signature). A worker thread runs generation; we decode the growing prefix
        (bit-exact, the codec is pure left-context) and emit the tail as frames arrive."""
        import queue
        import threading

        talker = self.model.model.talker
        st = self.model.model.speech_tokenizer
        fq = queue.Queue()

        def hook(_m, _i, out):
            cid = out.hidden_states[1]  # [1, 16] per frame, None during prefill
            if cid is not None:
                fq.put(cid.detach())

        handle = talker.register_forward_hook(hook)

        def worker():
            try:
                self.model.generate_custom_voice(
                    text=text, speaker=speaker, language=language,
                    do_sample=True, subtalker_dosample=True)
            except Exception as e:  # surface in the consumer
                fq.put(e)
            finally:
                handle.remove()
                fq.put(None)

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        frames, emitted, next_at = [], 0, chunk_frames
        while True:
            item = fq.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            frames.append(item)
            if len(frames) >= next_at:
                wavs, sr = st.decode([{"audio_codes": torch.cat(frames, dim=0)}])
                pcm = np.asarray(wavs[0], dtype=np.float32)
                next_at += chunk_frames
                if len(pcm) > emitted:
                    new, emitted = pcm[emitted:], len(pcm)
                    yield new, sr
        if frames:  # flush the tail
            wavs, sr = st.decode([{"audio_codes": torch.cat(frames, dim=0)}])
            pcm = np.asarray(wavs[0], dtype=np.float32)
            if len(pcm) > emitted:
                yield pcm[emitted:], sr
        th.join()
