# AudioClipsRemover
Python program to find unwanted audio (like specific commercials) and remove them from an audio file

## Background
I downloaded hundreds of episodes of a podcast only to find that whatever service (hopefully not the originating podcast creator) injected the same ad, sometimes multiple times, into most of the episodes. I wanted to find a way to remove these duplicated ads.

## Usage
(Optinally) create a virtual environment

### Windows
```
python -m venv .venv
.venv\Scripts\activate
```
### Linux/Mac?
```sh
python3 -m venv .venv
. .venv\bin\activate
```

### Example Run
Replace python with python3 if applicable
```
python audio_clips_remover.py -u Waterhose_commercial.arr.npy -u Waterhose_commercial2.arr.npy "2019-11-02 - Episode 93 - 5x05 Thirst.mp3"
```

This will create two directories: trimmed, discard, and place files in them

