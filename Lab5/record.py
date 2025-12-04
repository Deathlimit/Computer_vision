import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

def record_audio(filename="record.wav", duration=10, samplerate=44100):
    print(f"Запись началась... ({duration} секунд)")
    

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    
    sd.wait()  
    print("Запись завершена, сохраняю файл...")

    
    audio_int16 = (audio * 32767).astype(np.int16)
    wav.write(filename, samplerate, audio_int16)

    print(f"Аудио сохранено как {filename}")


if __name__ == "__main__":
    record_audio("input_audio.wav", duration=10)
