# Demo

Two ways to run the voice agent end to end. The pipeline is identical (STT -> LLM -> Qwen3-TTS ->
audio); only **where the talker decodes** and **how your mic/speaker reach it** change.

## Option A - local, one machine (simplest)

Runs entirely on one machine with a local mic and speaker.

```bash
cp .env.example .env          # set GROQ_API_KEY
python tts/demo.py            # reference backend (PyTorch), runs on any GPU/CPU
```

On a 5090, use the megakernel for the talker:
```bash
DECODE_BACKEND=megakernel python tts/demo.py
```

If the default audio device is wrong, set `MIC_DEVICE_INDEX` / `SPEAKER_DEVICE_INDEX` in `.env`.

## Option B - headless 5090 + laptop (the megakernel, live)

A cloud 5090 box is headless (no mic/speaker), and its outbound network can be flaky or block the
cloud WebRTC services (Daily, LiveKit) those demos usually rely on, which also need an account. So
instead of putting a transport on the box, **keep the box's network out of the loop**: run the voice
pipeline on your laptop and have only the **talker TTS** run on the 5090, reached over the SSH tunnel
you already have. Groq STT/LLM run on the laptop, so the box needs no internet during the demo (the
model is loaded locally).

```
laptop:  mic -> Groq STT -> Groq LLM --[HTTP over SSH tunnel]--> 5090: megakernel TTS
laptop:  speaker <----------------- PCM audio <-----------------/
```

**1. On the 5090** -- start the TTS server (loads the megakernel once, serves on :8090):
```bash
SKIP_CP_GRAPH=1 python tts/serve_tts.py     # prints "ready: POST /tts on :8090"
```
`SKIP_CP_GRAPH=1` is required: the CP CUDA-graph capture corrupts state on some 5090 instances
(generation never terminates; see the README). If the box's huggingface.co access is flaky, also
load from the local cache:
```bash
SKIP_CP_GRAPH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  QWEN_TTS_MODEL=/path/to/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/<hash> \
  python tts/serve_tts.py
```

**2. From your laptop** -- open an SSH tunnel to the server port:
```bash
ssh -N -L 8090:localhost:8090 -p <port> root@<box-ip>
```

**3. On your laptop** -- run the bridge (local mic/speaker + Groq, TTS over the tunnel):
```bash
cp .env.example .env               # set GROQ_API_KEY
python tts/demo_bridge.py          # POSTs text to http://localhost:8090/tts
```
It prints `Ready. Talk to it.` -- speak, and you hear the reply spoken by the megakernel on the 5090.

### Recording
Screen-record with **OBS** (add sources: Display Capture + Desktop Audio + Mic). Optionally show
`watch -n0.5 nvidia-smi` on the box in a second window; the GPU spikes each time it speaks, which
makes the 5090's role visible.

### Notes
- The bridge's TTS is **batch** (synthesize the whole reply, then stream it to the speaker): RTF
  **0.108** (`logs/bench_rtf.txt`); the recorded demo's live per-reply timing is in
  `logs/demo_serve.log` (RTF 0.115-0.123).
- `serve_tts.py` needs `fastapi` + `uvicorn` on the box; `demo_bridge.py` needs `requests` on the
  laptop. Both are in `requirements.txt`. The laptop does **not** need `torch` / `qwen-tts`.
- The speed/quality evidence is the benchmarks (`bench/`), the logs (`logs/`), and `samples/`; this
  demo shows the *integration* working: the talker on the megakernel, streaming into Pipecat.
