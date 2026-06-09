"""Laptop side of the live demo: mic -> Groq STT -> Groq LLM -> [megakernel TTS on the 5090] -> speaker.

The talker TTS runs on the 5090 via serve_tts.py, reached over an SSH tunnel (localhost:8080), so the
box's flaky outbound network stays out of the loop. Start the tunnel + serve_tts.py first. See DEMO.md.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

SAMPLE_RATE = 24000
TTS_URL = os.getenv("TTS_URL", "http://localhost:8090/tts")
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep replies short, plain and conversational - "
    "they are spoken aloud."
)


class RemoteTTSService(TTSService):
    """Sends text to the megakernel TTS server on the 5090 and streams back the audio."""

    def __init__(self, url=TTS_URL, speaker="Ryan", **kwargs):
        super().__init__(
            sample_rate=SAMPLE_RATE,
            settings=TTSSettings(model="qwen3-tts-12hz-0.6b", voice=speaker, language=None),
            **kwargs,
        )
        self._url = url
        self._speaker = speaker

    async def run_tts(self, text, context_id):
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, self._fetch, text)
        await self.stop_ttfb_metrics()
        step = int(SAMPLE_RATE * 0.04) * 2  # ~40 ms int16 frames
        for i in range(0, len(pcm), step):
            yield TTSAudioRawFrame(audio=pcm[i:i + step], sample_rate=SAMPLE_RATE, num_channels=1)
        yield TTSStoppedFrame()

    def _fetch(self, text):
        r = requests.post(self._url, json={"text": text, "speaker": self._speaker}, timeout=120)
        r.raise_for_status()
        return r.content


async def main():
    load_dotenv()
    api_key = os.environ["GROQ_API_KEY"]
    llm_model = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    stt_model = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

    devices = {}
    if os.getenv("MIC_DEVICE_INDEX"):
        devices["input_device_index"] = int(os.environ["MIC_DEVICE_INDEX"])
    if os.getenv("SPEAKER_DEVICE_INDEX"):
        devices["output_device_index"] = int(os.environ["SPEAKER_DEVICE_INDEX"])
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=True,
        audio_in_sample_rate=16000, audio_out_sample_rate=SAMPLE_RATE, **devices,
    ))

    stt = GroqSTTService(api_key=api_key, model=stt_model)
    llm = GroqLLMService(api_key=api_key, model=llm_model)
    tts = RemoteTTSService()
    print(f"TTS -> {TTS_URL} (megakernel on the 5090 over the SSH tunnel)")

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg,
    ])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True))
    runner = PipelineRunner(handle_sigint=False)
    print("Ready. Talk to it; Ctrl+C to stop.")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
