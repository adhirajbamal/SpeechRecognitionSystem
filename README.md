# SpeechRecognitionSystem
Python implementation of a speech-to-text system

# Speech Recognition System

A Python tool for converting speech to text using both online (Google Web Speech API) and offline (Wav2Vec 2.0) methods.

## Features

- Dual transcription modes:
  - Online: Google Web Speech API (requires internet)
  - Offline: Wav2Vec 2.0 transformer model
- Supports common audio formats (WAV, AIFF, FLAC)
- Interactive command-line interface
- Easy to extend with additional models

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/speech-recognition-system.git
   cd speech-recognition-system
2. Install dependencies:
   pip install -r requirements.txt


Usage
Run the speech recognition system: 
 python speech_to_text.py

 Follow the on-screen instructions to:

Choose transcription method

Provide path to audio file

View transcription results

Supported Audio Formats
WAV (recommended)

AIFF

FLAC

MP3 (requires ffmpeg)
