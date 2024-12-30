import sqlite3
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import PhotoImage
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
from PIL import Image, ImageTk
import textwrap

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


def clear_fields():
    name_entry.delete(0, tk.END)
    file_entry.delete(0, tk.END)


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

# Veritabanındaki verileri listelemek için fonksiyon
def list_data():
    for row in tree.get_children():
        tree.delete(row)

    conn = sqlite3.connect("audio_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audio_data")
    rows = cursor.fetchall()
    for row in rows:
        tree.insert("", tk.END, values=row)
    conn.close()

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


# Dalga formu ve spektrogram görüntüleme
def display_waveform_and_spectrogram():
    file_path = file_entry.get()
    if not file_path or not os.path.exists(file_path):
        messagebox.showwarning("Hata", "Lütfen geçerli bir ses dosyası seçin!")
        return

    # WAV dosyasını yükle
    sample_rate, data = wavfile.read(file_path)

    # Tek kanallı hale getir (mono)
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Spektrogram oluştur
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Zaman alanında göster
    axs[0].plot(data)
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("Amplitude")

    # Spektrogram
    axs[1].specgram(data, Fs=sample_rate, cmap="viridis")
    axs[1].set_title("Spectrogram")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")

    plt.colorbar(axs[1].images[0], ax=axs[1], orientation='horizontal', pad=0.2)
    plt.tight_layout()
    plt.show()



# Ses analizi için duygu analizi ekle
def analyze_emotion():
    file_path = file_entry.get()
    if not file_path or not os.path.exists(file_path):
        messagebox.showwarning("Hata", "Lütfen geçerli bir ses dosyası seçin!")
        return

    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Ses özelliklerini çıkar
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

        # Ortalama özellikleri al
        mfcc_mean = np.mean(mfcc.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        mel_mean = np.mean(mel.T, axis=0)

        # Özellikleri birleştir
        features = np.hstack((mfcc_mean, chroma_mean, mel_mean))

        # Veriyi ölçeklendir
        scaler = StandardScaler()
        features = scaler.fit_transform(features.reshape(1, -1))

        # Duygu analizi modeli (Burada örnek olarak rastgele yüzdelik değerler kullanıyoruz)
        # Gerçek duygu analizi için bir model kullanılabilir (örneğin, bir SVM veya NN)
        emotions = ["Mutlu", "Üzgün", "Öfkeli", "Korkmuş", "Sakin"]
        percentages = np.random.randint(10, 50, size=5)
        total = sum(percentages)
        percentages = [x / total * 100 for x in percentages]

        emotion_result = "\n".join([f"{emotion}: {percentage:.2f}%" for emotion, percentage in zip(emotions, percentages)])

        for widget in empty_frame.winfo_children():
            if widget != empty_label:  # Başlık widget'ını temizleme
                widget.destroy()

        # Sonuçları empty_frame içinde bir label olarak göster
        result_label = tk.Label(empty_frame, text=emotion_result, font=("Arial", 12), bg="#f5f5f5")
        result_label.pack(pady=10)

    except Exception as e:
        # Hata durumunda bir label ile mesaj göster
        error_label = tk.Label(empty_frame, text=f"Duygu analizi yapılamadı: {e}", font=("Arial", 12), fg="red", bg="#f5f5f5")
        error_label.pack(pady=10)


# Google Cloud Speech-to-Text API ile konuşmayı metne dönüştür
def speech_to_text():
    file_path = file_entry.get()
    if not file_path or not os.path.exists(file_path):
        messagebox.showwarning("Hata", "Lütfen geçerli bir ses dosyası seçin!")
        return

    try:
        # Google Cloud hizmet hesabı anahtarını ayarlayın
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sesprojesi-445723-69e8aecfc6f9.json"  # JSON dosyanızın yolu

        client = speech.SpeechClient()

        # Ses dosyasını yükle
        with io.open(file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="tr-TR",  # Türkçe için dil kodu
        )

        # Speech-to-Text işlemini gerçekleştir
        response = client.recognize(config=config, audio=audio)

        # Metni birleştir ve kelime sayısını hesapla
        transcript = " ".join(result.alternatives[0].transcript for result in response.results)
        word_count = len(transcript.split())

         # Önceki sonuçları temizle (varsa), başlık widget'ını bırak
        for widget in empty_frame.winfo_children():
            if widget != empty_label:  # Başlık widget'ını temizleme
                widget.destroy()

        # Metni satırlara bölme
        wrapper = textwrap.TextWrapper(width=65)  # 80 karakter genişliğinde satırlara böler
        word_list = wrapper.wrap(text=transcript)
        
        # Konuşmayı metne çevir sonuçlarını başlık altına satır satır yaz
        text_result = f"Metin: {' '.join(word_list)}\nKelime Sayısı: {word_count}"

        # Sonuçları her satırı ayrı bir label olarak yazdır
        for line in word_list:
            result_label = tk.Label(empty_frame, text=line, font=("Arial", 12), bg="#f5f5f5")
            result_label.pack(pady=2)

        # Kelime sayısını ekleyin
        word_count_label = tk.Label(empty_frame, text=f"Kelime Sayısı: {word_count}", font=("Arial", 12), bg="#f5f5f5")
        word_count_label.pack(pady=10)

    except Exception as e:
        # Hata durumunda bir label ile mesaj göster
        error_label = tk.Label(empty_frame, text=f"Speech-to-Text işlemi başarısız: {e}", font=("Arial", 12), fg="red", bg="#f5f5f5")
        error_label.pack(pady=10)




def analyze_topic():
    file_path = file_entry.get()
    if not file_path or not os.path.exists(file_path):
        messagebox.showwarning("Hata", "Lütfen geçerli bir ses dosyası seçin!")
        return

    try:
        # Google Cloud hizmet hesabı anahtarını ayarlayın
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sesprojesi-445723-69e8aecfc6f9.json"  # JSON dosyanızın yolu

        # Speech-to-Text işlemi
        speech_client = speech.SpeechClient()
        with io.open(file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="tr-TR",  # Türkçe için dil kodu
        )
        response = speech_client.recognize(config=config, audio=audio)
        transcript = " ".join(result.alternatives[0].transcript for result in response.results)

        # Metni İngilizce'ye çevir
        translate_client = translate.Client()
        translation = translate_client.translate(transcript, target_language="en")
        translated_text = translation['translatedText']

        # Konu analizi
        language_client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=translated_text, type_=language_v1.Document.Type.PLAIN_TEXT)
        classification_response = language_client.classify_text(request={"document": document})

        # Sonuçları göster
        results = []
        for category in classification_response.categories:
            results.append(f"Kategori: {category.name}, Güven: {category.confidence:.2f}")

        
        # Önceki sonuçları temizle (varsa), başlık widget'ını bırak
        for widget in empty_frame.winfo_children():
            if widget != empty_label:  # Başlık widget'ını temizleme
                widget.destroy()

        # Kategori sonuçlarını yazdır
        if results:
            for result in results:
                result_label = tk.Label(empty_frame, text=result, font=("Arial", 12), bg="#f5f5f5")
                result_label.pack(pady=2)
        else:
            result_label = tk.Label(empty_frame, text="Konu sınıflandırması yapılamadı!", font=("Arial", 12), bg="#f5f5f5")
            result_label.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Hata", f"Konu analizi başarısız: {e}")



# Modelden tahmin yapma fonksiyonu
def predict_person(file_path, model, scaler):
    features = extract_features(file_path).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return prediction[0]



# Ses özelliklerini çıkarma fonksiyonu
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    features = np.hstack((np.mean(mfcc.T, axis=0), np.mean(chroma.T, axis=0), np.mean(mel.T, axis=0)))
    return features

# Veritabanındaki ses dosyalarını ve sahip isimlerini al
def load_training_data():
    conn = sqlite3.connect("audio_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT owner_name, file_path FROM audio_data")
    data = cursor.fetchall()
    conn.close()

    features = []
    labels = []
    for owner_name, file_path in data:
        if os.path.exists(file_path):
            features.append(extract_features(file_path))
            labels.append(owner_name)
    return np.array(features), np.array(labels)

# Modeli eğit
def train_model():
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performans metrikleri
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    messagebox.showinfo("Model Eğitimi Tamamlandı", f"Accuracy: {acc:.2f}\nF1-Score: {f1:.2f}")
    return model

# Modeli yükle ve anlık ses tanıma yap
trained_model = None

def recognize_from_recording():
    global trained_model
    trained_model = train_model()
    file_path = file_entry.get()
    if not file_path or not os.path.exists(file_path):
        messagebox.showwarning("Hata", "Lütfen geçerli bir ses dosyası seçin!")
        return

    features = extract_features(file_path).reshape(1, -1)
    prediction = trained_model.predict(features)
    messagebox.showinfo("Kişi Tanıma Sonucu", f"Tahmin Edilen Kişi: {prediction[0]}")







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



# Ana pencereyi oluştur
app = tk.Tk()
app.title("Ses ve Sahip Kaydı")
app.geometry(f"{app.winfo_screenwidth()}x{app.winfo_screenheight()}")
app.configure(bg="#222C35")

# Başlık
header = tk.Label(app, text="Ses ve Sahip Kaydı", font=("Arial", 24, "bold"), bg="#222C35", fg="white", pady=10)
header.pack(fill=tk.X)

# Ana çerçeve
main_frame = tk.Frame(app, bg="#222C35")
main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

# Sol çerçeve
left_frame = tk.Frame(main_frame, bg="#222C35", bd=2, relief="solid")
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

left_label = tk.Label(left_frame, text="Bilgi Girişi", font=("Arial", 16, "bold"), bg="#222C35",fg="white")
left_label.pack(pady=10)

# Giriş alanları
name_label = tk.Label(left_frame, text="Sahibin Adı:", font=("Arial", 12), bg="#222C35", fg="white")
name_label.pack(pady=5,padx=27,  anchor="w")
name_entry = tk.Entry(left_frame, width=40, font=("Arial", 12))
name_entry.pack(pady=5, padx=30)

file_label = tk.Label(left_frame, text="Ses Dosyası:", font=("Arial", 12), bg="#222C35",fg="white")
file_label.pack(pady=5,padx=27, anchor="w")
file_frame = tk.Frame(left_frame, bg="#222C35")
file_frame.pack(pady=5)
file_entry = tk.Entry(file_frame, width=30, font=("Arial", 12))
file_entry.pack(side=tk.LEFT, padx=5)
file_button = tk.Button(file_frame, text="Seç", command=select_file, bg="white", fg="black", font=("Arial", 12), relief="flat")
file_button.pack(side=tk.LEFT)

# Çöp kutusu simgesini yeniden boyutlandır
original_icon = Image.open("trash_icon.png")
resized_icon = original_icon.resize((24, 24), Image.Resampling.LANCZOS)  # 24x24 boyutunda
trash_icon = ImageTk.PhotoImage(resized_icon)

clear_button = tk.Button(file_frame, image=trash_icon, command=clear_fields, bg="#ffffff", relief="flat")
clear_button.pack(side=tk.LEFT, padx=5)

# Kaydet butonu
save_button = tk.Button(left_frame, text="Kaydet", command=save_data, bg="red", fg="white", font=("Arial", 12), relief="flat")
save_button.pack(pady=10,padx=5)

# Listeleme butonu
list_button = tk.Button(left_frame, text="Listele", command=list_data, bg="blue", fg="white", font=("Arial", 12), relief="flat")
list_button.pack(pady=10, padx=5)

# Sağ çerçeve
right_frame = tk.Frame(main_frame, bg="#222C35", bd=2, relief="solid")
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

right_label = tk.Label(right_frame, text="İşlemler", font=("Arial", 16, "bold"), bg="#222C35", fg="white")
right_label.pack(pady=10)

# İşlem butonları
record_button = tk.Button(right_frame, text="Kayda Başla", command=start_recording, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=17)
record_button.pack(pady=5)

stop_button = tk.Button(right_frame, text="Kaydı Durdur", command=stop_recording, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=17)
stop_button.pack(pady=5, )

play_button = tk.Button(right_frame, text="Ses Dosyasını Oynat", command=play_audio, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=17)
play_button.pack(pady=5, )

stop_play_button = tk.Button(right_frame, text="Ses Oynatmayı Durdur", command=stop_audio, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=17)
stop_play_button.pack(pady=5,padx=15)

# Ek işlemler çerçevesi
extra_frame = tk.Frame(main_frame, bg="#222C35", bd=2, relief="solid")
extra_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

extra_label = tk.Label(extra_frame, text="Ek İşlemler", font=("Arial", 16, "bold"), bg="#222C35",fg="white")
extra_label.pack(pady=10)

plot_button = tk.Button(extra_frame, text="Dalga Formu ve Spektrogram Göster", command=display_waveform_and_spectrogram, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=30)
plot_button.pack(pady=10,)

emotion_button = tk.Button(extra_frame, text="Ses Analizi (Duygu)", command=analyze_emotion, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=30)
emotion_button.pack(pady=5, )

speech_to_text_button = tk.Button(extra_frame, text="Konuşmayı Metne Çevir ve Kelime Say", command=speech_to_text, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=30)
speech_to_text_button.pack(pady=5, padx=15)

topic_analysis_button = tk.Button(extra_frame, text="Konuşulan Konuyu Analiz Et", command=analyze_topic, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=30)
topic_analysis_button.pack(pady=5,)

realtime_button = tk.Button(extra_frame, text="Anlık Ses Tanıma", command=recognize_from_recording, bg="#519ABA", fg="white", font=("Arial", 12), relief="flat",width=30)
realtime_button.pack(pady=5, )


# Ekstra çerçeve düzenlemeleri (kalan alanı dolduracak şekilde)
empty_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief="solid")
empty_frame.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

# Sonuçlar başlığı
empty_label = tk.Label(empty_frame, text="Sonuçlar", font=("Arial", 16, "bold"), bg="#ffffff")
empty_label.pack(pady=10)

# Bu çerçeve tüm boş alanı dolduracak şekilde
main_frame.grid_rowconfigure(0, weight=1)  # İlk satırın esnek olmasını sağla
main_frame.grid_columnconfigure(3, weight=1)  # Dördüncü sütunun esnek olmasını sağla


# Listeleme alanı (Treeview)
columns = ("ID", "Sahip Adı", "Dosya Yolu")
tree = ttk.Treeview(app, columns=columns, show="headings")
tree.heading("ID", text="ID")
tree.heading("Sahip Adı", text="Sahip Adı")
tree.heading("Dosya Yolu", text="Dosya Yolu")

style = ttk.Style()
style.configure("Treeview", font=("Arial", 12), rowheight=25)
style.configure("Treeview.Heading", font=("Arial", 14), background="#4CAF50", foreground="black")

tree.pack(pady=10, fill=tk.BOTH, expand=True)

# Veritabanını oluştur
create_database()

# Tkinter döngüsünü başlat
app.mainloop()