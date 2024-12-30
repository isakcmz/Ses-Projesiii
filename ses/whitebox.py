import unittest
import os
import sqlite3
import numpy as np
from unittest.mock import patch, MagicMock
from sestanimaprojesi import (
    create_database,
    save_to_database,
    load_training_data,
    record_audio,
    play_audio,
    stop_audio,
    train_model,
    extract_features
)

class TestAudioApplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Veritabanı ve dosya yapısını hazırla."""
        create_database()
        cls.test_db = "audio_database.db"

    @classmethod
    def tearDownClass(cls):
        """Testler bittiğinde kaynakları temizle."""
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_create_database(self):
        """Veritabanı oluşturma işlemini test et."""
        self.assertTrue(os.path.exists(self.test_db), "Veritabanı oluşturulamadı.")

    def test_save_to_database(self):
        """Veritabanına ses dosyası ve sahibi kaydetme işlemini test et."""
        owner_name = "Test Owner"
        file_path = "test_audio.wav"
        save_to_database(owner_name, file_path)
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audio_data WHERE owner_name=?", (owner_name,))
        rows = cursor.fetchall()
        conn.close()
        self.assertGreater(len(rows), 0, "Kayıt veritabanına eklenemedi.")

    def test_load_training_data(self):
        """Ses dosyalarının ve özelliklerin başarıyla yüklenip yüklenmediğini test et."""
        features, labels = load_training_data()
        self.assertTrue(isinstance(features, np.ndarray), "Özellikler ndarray değil.")
        self.assertTrue(isinstance(labels, np.ndarray), "Etiketler ndarray değil.")

    @patch("main_code.sd.InputStream")
    def test_record_audio(self, mock_input_stream):
        """Ses kaydını test et."""
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream
        file_path = "test_recording.wav"
        record_audio(file_path)
        self.assertTrue(os.path.exists(file_path), "Ses kaydedilemedi.")
        os.remove(file_path)

    @patch("main_code.pygame.mixer")
    def test_play_audio(self, mock_mixer):
        """Ses oynatma işlemini test et."""
        mock_mixer.init.return_value = None
        mock_mixer.music.load.return_value = None
        mock_mixer.music.play.return_value = None
        file_path = "test_audio.wav"
        with open(file_path, "wb") as f:
            f.write(b"dummy audio data")
        play_audio(file_path)
        mock_mixer.music.play.assert_called_once()
        os.remove(file_path)

    @patch("main_code.pygame.mixer")
    def test_stop_audio(self, mock_mixer):
        """Ses durdurma işlemini test et."""
        mock_mixer.music.stop.return_value = None
        stop_audio()
        mock_mixer.music.stop.assert_called_once()

    def test_train_model(self):
        """Makine öğrenimi modelinin eğitilmesini test et."""
        with patch("main_code.RandomForestClassifier.fit") as mock_fit:
            mock_fit.return_value = None
            model = train_model()
            self.assertIsNotNone(model, "Model eğitimi başarısız oldu.")

    def test_extract_features(self):
        """Ses dosyasından özellik çıkarma işlemini test et."""
        with patch("main_code.librosa.load") as mock_load:
            mock_load.return_value = (np.random.rand(22050), 22050)  # 1 saniye süren dummy ses
            features = extract_features("dummy_path.wav")
            self.assertEqual(features.shape[0], 40, "Çıkarılan özellik sayısı yanlış.")

if __name__ == "__main__":
    unittest.main()
