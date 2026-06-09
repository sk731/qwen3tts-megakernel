"""End-to-end voice demo: STT -> LLM -> Qwen3-TTS -> audio.

DECODE_BACKEND=reference|megakernel; set DAILY_ROOM_URL for a headless (Daily) demo. See DEMO.md.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
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

from tts.pipecat_service import Qwen3TTSService

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep replies short, plain and conversational - "
    "they are spoken aloud."
)


def make_backend():
    if os.getenv("DECODE_BACKEND", "reference").lower() == "megakernel":
        from decoders.megakernel import MegakernelTalker  # imports only on a 5090
        print("backend: megakernel (RTX 5090)")
        return MegakernelTalker(accel_codepred=True)
    from decoders.reference import ReferenceTalker
    print("backend: reference (PyTorch)")
    return ReferenceTalker()


def make_transport():
    lk_url = os.getenv("LIVEKIT_URL")
    if lk_url:
        from livekit import api
        from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
        room = os.getenv("LIVEKIT_ROOM", "megakernel")
        # the bot mints its own join token from the project key/secret
        token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
                 .with_identity("qwen-tts-bot").with_name("Qwen3-TTS bot")
                 .with_grants(api.VideoGrants(room_join=True, room=room)).to_jwt())
        print(f"transport: LiveKit ({lk_url} room={room}) - join this room from your browser")
        return LiveKitTransport(
            lk_url, token, room,
            LiveKitParams(
                audio_in_enabled=True, audio_out_enabled=True,
                audio_in_sample_rate=16000, audio_out_sample_rate=24000,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )
    room = os.getenv("DAILY_ROOM_URL")
    if room:
        from pipecat.transports.daily.transport import DailyParams, DailyTransport
        print(f"transport: Daily ({room}) - join this room from your browser")
        return DailyTransport(
            room, os.getenv("DAILY_TOKEN"), "Qwen3-TTS bot",
            DailyParams(
                audio_in_enabled=True, audio_out_enabled=True,
                audio_in_sample_rate=16000, audio_out_sample_rate=24000,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    # the system default device may be a virtual cable; pin real ones via .env
    # (MIC_DEVICE_INDEX / SPEAKER_DEVICE_INDEX), find indices with scripts/mic_check.py
    devices = {}
    if os.getenv("MIC_DEVICE_INDEX"):
        devices["input_device_index"] = int(os.environ["MIC_DEVICE_INDEX"])
    if os.getenv("SPEAKER_DEVICE_INDEX"):
        devices["output_device_index"] = int(os.environ["SPEAKER_DEVICE_INDEX"])
    print("transport: local mic/speaker")
    return LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=True,
        audio_in_sample_rate=16000, audio_out_sample_rate=24000, **devices,
    ))


async def main():
    load_dotenv()
    api_key = os.environ["GROQ_API_KEY"]
    # Groq model names change over time; override via .env if these are deprecated.
    llm_model = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    stt_model = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

    transport = make_transport()
    stt = GroqSTTService(api_key=api_key, model=stt_model)
    llm = GroqLLMService(api_key=api_key, model=llm_model)
    tts = Qwen3TTSService(backend=make_backend())

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True))
    runner = PipelineRunner(handle_sigint=False)
    print("Ready. Talk to it; Ctrl+C to stop.")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
