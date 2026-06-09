"""HTTP TTS server on the 5090: POST /tts {text} -> raw 24kHz mono PCM16. Megakernel, no internet.

Lets the voice pipeline run on a laptop (mic + Groq STT/LLM) and reach the megakernel over an SSH
tunnel, so a headless box's outbound network stays out of the loop. Pair with demo_bridge.py. See DEMO.md.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from decoders.megakernel import MegakernelTalker

print("loading megakernel talker ...", flush=True)
mk = MegakernelTalker(accel_codepred=True)  # __init__ does a safe codec warmup
print("ready: POST /tts on :8090", flush=True)

app = FastAPI()


class Req(BaseModel):
    text: str
    speaker: str = "Ryan"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/tts")
def tts(r: Req):
    t = time.perf_counter()
    wav, sr, _ = mk.synthesize(r.text, r.speaker, greedy=False)
    dt = time.perf_counter() - t
    audio_s = len(wav) / sr
    print(f"[TTS] megakernel: {dt * 1000:6.0f} ms  audio {audio_s:5.2f}s  "
          f"RTF {dt / audio_s:.3f}  | {r.text[:60]!r}", flush=True)
    pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return Response(content=pcm, media_type="application/octet-stream",
                    headers={"x-sample-rate": str(sr)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="warning")
