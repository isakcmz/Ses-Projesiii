# Ses özelliklerini çıkarma fonksiyonu
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    features = np.hstack((np.mean(mfcc.T, axis=0), np.mean(chroma.T, axis=0), np.mean(mel.T, axis=0)))
    return features

# Verileri bir klasörden yükleme fonksiyonu
def load_data_from_directory(directory):
    features, labels = [], []
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            for file_name in os.listdir(person_dir):
                file_path = os.path.join(person_dir, file_name)
                if file_name.endswith(('.wav', '.mp3', '.aac', '.ogg')):
                    try:
                        features.append(extract_features(file_path))
                        labels.append(person_name)
                    except Exception as e:
                        print(f"Hata: {file_path} dosyası işlenemedi. {e}")
    return np.array(features), np.array(labels)

# Model eğitimi fonksiyonu
def train_model(directory):
    features, labels = load_data_from_directory(directory)
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Yeterli veri bulunamadı!")

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    messagebox.showinfo("Model Eğitimi Tamamlandı", f"Accuracy: {acc:.2f}\nF1-Score: {f1:.2f}")

    return model, scaler

# Modelden tahmin yapma fonksiyonu
def predict_person(file_path, model, scaler):
    features = extract_features(file_path).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return prediction[0]

# Tkinter arayüzü
class AudioApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Ses ve Sahip Kaydı")
        self.root.geometry("800x600")

        self.model = None
        self.scaler = None

        self.setup_ui()

    def setup_ui(self):
        self.directory_label = tk.Label(self.root, text="Veri Klasörü:")
        self.directory_label.pack(pady=5)

        self.directory_entry = tk.Entry(self.root, width=50)
        self.directory_entry.pack(pady=5)

        self.browse_button = tk.Button(self.root, text="Klasör Seç", command=self.browse_directory)
        self.browse_button.pack(pady=5)

        self.train_button = tk.Button(self.root, text="Modeli Eğit", command=self.train_model)
        self.train_button.pack(pady=10)

        self.file_label = tk.Label(self.root, text="Tahmin Edilecek Ses Dosyası:")
        self.file_label.pack(pady=5)

        self.file_entry = tk.Entry(self.root, width=50)
        self.file_entry.pack(pady=5)

        self.file_button = tk.Button(self.root, text="Dosya Seç", command=self.select_file)
        self.file_button.pack(pady=5)

        self.predict_button = tk.Button(self.root, text="Kişiyi Tahmin Et", command=self.predict_person)
        self.predict_button.pack(pady=10)

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.directory_entry.delete(0, tk.END)
            self.directory_entry.insert(0, directory)

    def train_model(self):
        directory = self.directory_entry.get()
        if not os.path.exists(directory):
            messagebox.showerror("Hata", "Geçerli bir klasör seçin!")
            return

        try:
            self.model, self.scaler = train_model(directory)
        except Exception as e:
            messagebox.showerror("Hata", f"Model eğitimi başarısız: {e}")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Ses Dosyasını Seç",
            filetypes=[("Ses Dosyaları", ".mp3;.wav;.aac;.ogg")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def predict_person(self):
        if not self.model or not self.scaler:
            messagebox.showerror("Hata", "Lütfen önce modeli eğitin!")
            return

        file_path = self.file_entry.get()
        if not os.path.exists(file_path):
            messagebox.showerror("Hata", "Geçerli bir ses dosyası seçin!")
            return

        try:
            prediction = predict_person(file_path, self.model, self.scaler)
            messagebox.showinfo("Tahmin Sonucu", f"Tahmin Edilen Kişi: {prediction}")
        except Exception as e:
            messagebox.showerror("Hata", f"Tahmin işlemi başarısız: {e}")

if _name_ == "_main_":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()

def recognize_from_recording():
    if not app.model or not app.scaler:
        messagebox.showerror("Hata", "Lütfen önce modeli eğitin!")
        return

    file_path = app.file_entry.get()
    if not os.path.exists(file_path):
        messagebox.showerror("Hata", "Geçerli bir ses dosyası seçin!")
        return

    try:
        prediction = predict_person(file_path, app.model, app.scaler)
        messagebox.showinfo("Tahmin Sonucu", f"Tahmin Edilen Kişi: {prediction}")
    except Exception as e:
        messagebox.showerror("Hata", f"Tahmin işlemi başarısız: {e}")