"""Backend interface shared by the reference and megakernel decoders."""

from abc import ABC, abstractmethod


class TalkerDecoder(ABC):
    @abstractmethod
    def synthesize(self, text, speaker, language="English", greedy=True):
        """text -> (wav float32, sample_rate, talker_tokens) for one utterance."""

    @abstractmethod
    def stream(self, text, speaker, language="English", chunk_frames=4):
        """Yield (pcm float32, sample_rate) chunks as codec frames decode."""
