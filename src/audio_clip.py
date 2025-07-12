from typing import Self
import soundfile as sf

class AudioClip():
    @staticmethod
    def from_audio_file(file_path : str) -> Self:
        audio_data, sample_rate = sf.read(file_path)
        audio_clip = AudioClip(audio_data, sample_rate)
        return audio_clip

    def __init__(self, audio_data, sample_rate: int):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.audio_mean = None # TODO: write test for and calculate this useful representation of the data

    @property
    def clip_length(self):
        pass

    def downmix_stereo_to_mono(self) -> Self:
        pass

    def remove_audio_clip(self, start_index, end_index, capture_discard=False) -> tuple[Self, Self]:
        """
        Returns a Tuple of (audio clip with segment removed, removed segment)
        """
        pass

    def trim_whitespace(self) -> Self:
        pass

    def save(self, path):
        pass

    def has_matching_sample_rate(self, other_clip: Self) -> bool:
        pass

    def prepend(self, other_clip: Self) -> Self:
        """
        Used especially for building up a discarded audio file
        """
        pass
