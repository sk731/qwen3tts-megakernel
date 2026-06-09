"""Pipecat TTS service: streams TTSAudioRawFrame chunks from any TalkerDecoder backend."""

import asyncio

import numpy as np
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from decoders.reference import ReferenceTalker

SAMPLE_RATE = 24000


def _pcm_to_bytes(pcm):
    # float32 [-1, 1] -> little-endian int16 PCM bytes
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Qwen3TTSService(TTSService):
    def __init__(self, backend=None, speaker="Ryan", chunk_frames=4, **kwargs):
        # store-mode settings must be fully initialised; None for what we don't use,
        # language=None avoids the string->Language enum conversion in the base.
        super().__init__(
            sample_rate=SAMPLE_RATE,
            settings=TTSSettings(model="qwen3-tts-12hz-0.6b", voice=speaker, language=None),
            **kwargs,
        )
        self._backend = backend or ReferenceTalker()
        self._speaker = speaker
        self._chunk_frames = chunk_frames

    async def run_tts(self, text, context_id):
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        gen = self._backend.stream(text, speaker=self._speaker, chunk_frames=self._chunk_frames)
        loop = asyncio.get_running_loop()
        first = True
        while True:
            item = await loop.run_in_executor(None, lambda: next(gen, None))
            if item is None:
                break
            pcm, sr = item
            if first:
                await self.stop_ttfb_metrics()  # first audio out = TTFC
                first = False
            yield TTSAudioRawFrame(audio=_pcm_to_bytes(pcm), sample_rate=sr, num_channels=1)

        yield TTSStoppedFrame()
