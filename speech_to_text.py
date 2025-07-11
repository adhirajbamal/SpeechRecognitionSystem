import os
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import numpy as np

class SpeechRecognizer:
    def __init__(self):
        """
        Initialize speech recognizer with both online (Google) and offline (Wav2Vec) options
        """
        self.recognizer = sr.Recognizer()
        
        # Load Wav2Vec model (will be loaded on first use)
        self.wav2vec_model = None
        self.wav2vec_processor = None
    
    def transcribe_with_google(self, audio_file_path):
        """
        Transcribe using Google Web Speech API (requires internet)
        :param audio_file_path: path to audio file
        :return: transcribed text
        """
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            return f"Google Speech Recognition error: {str(e)}"
    
    def load_wav2vec(self):
        """
        Lazy loading of Wav2Vec model to save memory
        """
        if self.wav2vec_model is None:
            model_name = "facebook/wav2vec2-base-960h"
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    def transcribe_with_wav2vec(self, audio_file_path):
        """
        Transcribe using Wav2Vec model (works offline)
        :param audio_file_path: path to audio file
        :return: transcribed text
        """
        try:
            self.load_wav2vec()
            
            # Load audio file
            speech_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
            
            # Process with Wav2Vec
            inputs = self.wav2vec_processor(
                speech_array, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                logits = self.wav2vec_model(inputs.input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]
            
            return transcription
        except Exception as e:
            return f"Wav2Vec transcription error: {str(e)}"

def main():
    print("CODTECH SPEECH RECOGNITION SYSTEM")
    print("---------------------------------\n")
    
    recognizer = SpeechRecognizer()
    
    while True:
        print("\nOptions:")
        print("1. Transcribe using Google Web Speech API (requires internet)")
        print("2. Transcribe using Wav2Vec (offline)")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ")
        
        if choice == '3':
            print("Exiting speech recognition system...")
            break
            
        if choice in ('1', '2'):
            audio_path = input("Enter path to audio file: ")
            
            if not os.path.exists(audio_path):
                print("File not found. Please try again.")
                continue
                
            if choice == '1':
                print("\nTranscribing with Google Web Speech API...")
                result = recognizer.transcribe_with_google(audio_path)
            else:
                print("\nTranscribing with Wav2Vec (offline)...")
                result = recognizer.transcribe_with_wav2vec(audio_path)
            
            print("\nTranscription Result:")
            print("--------------------")
            print(result)
            print("\n")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
