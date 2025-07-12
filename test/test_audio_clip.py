from unittest.mock import patch
import numpy as np
from audio_clip import AudioClip

import soundfile as sf # You'll need to install this library: pip install soundfile

def test_should_create_an_audio_clip_from_audio_file_and_store_as_np_array():
    # Arrange
    fake_audio_file = np.array([1, 2, 3])
    fake_sample_rate = 48000

    with patch.object(sf, 'read') as mock_read:
        # We assume that the Soundfile library is working correctly. We mock out it's 'read' behavior to make the test run faster, knowing that it always returns an NP array and an integer
        mock_read.return_value = (fake_audio_file, fake_sample_rate)
        input_file_name = "sample.mp3"

        # Act
        audio_clip = AudioClip.from_audio_file(input_file_name)

        # Assert
        mock_read.assert_called_with(input_file_name)

    assert (audio_clip.audio_data == fake_audio_file).all()
    assert audio_clip.sample_rate == fake_sample_rate
