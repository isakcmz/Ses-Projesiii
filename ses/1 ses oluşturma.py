import sounddevice as sd
import numpy as np
import wave

# Ses kaydedeceğimiz parametreler
samplerate = 44100  # 44.1 kHz
duration = 10  # saniye
channels = 1  # mono ses kaydı

# Ses kaydını başlat
audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='int16')
sd.wait()  # Kayıt tamamlanana kadar bekle

# Kaydı WAV dosyası olarak kaydet
file_path = "konusma.wav"
with wave.open(file_path, 'wb') as wf:
    wf.setnchannels(channels)  # Mono
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(samplerate)  # 44.1 kHz
    wf.writeframes(audio_data.tobytes())  # Veriyi yaz

print("Ses kaydı başarıyla oluşturuldu: konusma.wav")
