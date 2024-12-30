import sqlite3
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import sounddevice as sd
import wave
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import librosa
import pygame  # Ses oynatma ve durdurma için kullanılır
import librosa.display
from sklearn.preprocessing import StandardScaler
from google.cloud import speech
from google.cloud import language_v1
import io
from google.cloud import translate_v2 as translate

# Veritabanını oluştur ve tabloyu tanımla
def create_database():
    conn = sqlite3.connect("audio_database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audio_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_name TEXT NOT NULL,
            file_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Ses dosyasını ve sahibi adını veritabanına kaydet
def save_to_database(owner_name, file_path):
    conn = sqlite3.connect("audio_database.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO audio_data (owner_name, file_path) VALUES (?, ?)", (owner_name, file_path))
    conn.commit()
    conn.close()


# Ses dosyasını seçmek için fonksiyon
def select_file():
    file_path = filedialog.askopenfilename(
        title="Ses Dosyasını Seç",
        filetypes=[("Ses Dosyaları", "*.mp3;*.wav;*.aac;*.ogg")]
    )
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

# Kaydet butonunun işlevi
def save_data():
    owner_name = name_entry.get()
    file_path = file_entry.get()

    if not owner_name or not file_path:
        messagebox.showerror("Hata", "Lütfen tüm alanları doldurun!")
        return

    if not os.path.exists(file_path):
        messagebox.showerror("Hata", "Seçilen dosya bulunamadı!")
        return

    save_to_database(owner_name, file_path)
    messagebox.showinfo("Başarılı", "Veriler başarıyla kaydedildi!")
    name_entry.delete(0, tk.END)
    file_entry.delete(0, tk.END)


# Ses kaydı için değişkenler
is_recording = False
output_file = ""

# Ses kaydını başlat
def start_recording():
    global is_recording, output_file
    owner_name = name_entry.get()
    if not owner_name:
        messagebox.showerror("Hata", "Lütfen sahibin adını girin!")
        return

    output_file = f"{owner_name}_recording.wav"
    is_recording = True
    threading.Thread(target=record_audio, args=(output_file,)).start()
    messagebox.showinfo("Bilgi", "Ses kaydı başladı!")

# Ses kaydını durdur
def stop_recording():
    global is_recording
    is_recording = False
    messagebox.showinfo("Bilgi", "Ses kaydı durduruldu ve kaydedildi!")
    file_entry.delete(0, tk.END)
    file_entry.insert(0, output_file)

# Ses kaydı işlevi
def record_audio(file_path):
    global is_recording
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(44100)  # 44.1kHz
        with sd.InputStream(samplerate=44100, channels=1, dtype="int16") as stream:
            while is_recording:
                data = stream.read(1024)[0]
                wf.writeframes(data)


# Ses dosyasını oynat
def play_audio():
    file_path = file_entry.get()
    if not file_path or not os.path.exists(file_path):
        messagebox.showwarning("Hata", "Lütfen geçerli bir ses dosyası seçin!")
        return

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Hata", f"Ses dosyası oynatılamadı: {e}")

# Ses oynatmayı durdur
def stop_audio():
    try:
        pygame.mixer.music.stop()
    except Exception as e:
        messagebox.showerror("Hata", f"Ses durdurulamadı: {e}")



# Tkinter arayüzü oluştur
app = tk.Tk()
app.title("Ses ve Sahip Kaydı")
app.geometry(f"{app.winfo_screenwidth()}x{app.winfo_screenheight()}")

# Etiketler ve giriş alanları
name_label = tk.Label(app, text="Sahibin Adı:")
name_label.pack(pady=5)
name_entry = tk.Entry(app, width=50)
name_entry.pack(pady=5)

file_label = tk.Label(app, text="Ses Dosyası:", bg="lightgray")
file_label.pack(pady=5)
file_frame = tk.Frame(app)
file_frame.pack(pady=5)
file_entry = tk.Entry(file_frame, width=40)
file_entry.pack(side=tk.LEFT, padx=5)
file_button = tk.Button(file_frame, text="Seç", command=select_file)
file_button.pack(side=tk.LEFT)

# Kaydet butonu
save_button = tk.Button(app, text="Kaydet", command=save_data)
save_button.pack(pady=10)

# Ses kaydı butonları
record_button = tk.Button(app, text="Kayda Başla", command=start_recording)
record_button.pack(pady=5)
stop_button = tk.Button(app, text="Kaydı Durdur", command=stop_recording)
stop_button.pack(pady=5)

# Ses dosyasını oynat ve durdur butonları
play_button = tk.Button(app, text="Ses Dosyasını Oynat", command=play_audio)
play_button.pack(pady=10)
stop_play_button = tk.Button(app, text="Ses Oynatmayı Durdur", command=stop_audio)
stop_play_button.pack(pady=10)

# Veritabanını oluştur
create_database()

# Tkinter döngüsünü başlat
app.mainloop()