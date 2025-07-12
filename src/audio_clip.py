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