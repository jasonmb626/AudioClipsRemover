from pathlib import Path
import sys
from unittest.mock import patch
from audio_clips_remover import main


def test_should_remove_ads_from_short_clip():
    input_file_name = "2019-11-02 - Episode 93 - 5x05 Thirst-trimmed.mp3"
    with patch.object(sys, 'argv', ['-u', 'Waterhose_commercial.arr.npy', '-u', 'Waterhose_commercial2.arr.npy', input_file_name]):
        main()
        actual_output_file = (Path('trimmed') / input_file_name)
        assert actual_output_file.exists()
        assert actual_output_file.read_bytes() == (Path('test') / 'resources'/ input_file_name ).read_bytes()
