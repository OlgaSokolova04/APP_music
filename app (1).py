from flask import Flask, request, render_template, jsonify
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Параметры модели (из вашего кода)
SR = 11025
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40
FMIN = 20
FMAX = 5000
FRAME_LENGTH = 256
BPM_RANGE = range(30, 286)
NUM_CLASSES = len(BPM_RANGE)

# Загрузка модели
model = tf.keras.models.load_model('model.h5')

# Инициализация LabelEncoder
le = LabelEncoder()
le.fit(list(BPM_RANGE))

# Функция извлечения Mel-спектограммы (из вашего кода)
def extract_mel_spectrogram(file_path, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        required_samples = FRAME_LENGTH * HOP_LENGTH
        if len(y) < required_samples:
            y = np.pad(y, (0, max(0, required_samples - len(y))), mode='constant')
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, window='hamming'
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db, 1.0
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None, 1.0

# Функция для получения окон спектограммы
def get_spectrogram_windows(mel_spec, frame_length=FRAME_LENGTH, hop_size=128):
    windows = []
    for start in range(0, mel_spec.shape[1] - frame_length + 1, hop_size):
        window = mel_spec[:, start:start + frame_length]
        if window.shape[1] == frame_length:
            windows.append(window)
    return np.array(windows)

# Функция для оценки глобального темпа
def estimate_global_tempo(model, mel_spec, le):
    windows = get_spectrogram_windows(mel_spec)
    if len(windows) == 0:
        return None
    windows = windows[..., np.newaxis]
    windows = (windows - windows.min()) / (windows.max() - windows.min())
    predictions = model.predict(windows)
    avg_predictions = np.mean(predictions, axis=0)
    tempo_class = np.argmax(avg_predictions)
    return le.inverse_transform([tempo_class])[0]

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Обработка загрузки файла и предсказания
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Сохранение загруженного файла
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Извлечение Mel-спектограммы
    mel_spec, _ = extract_mel_spectrogram(file_path)
    if mel_spec is None:
        return jsonify({'error': 'Error processing audio file'}), 500

    # Предсказание темпа
    tempo = estimate_global_tempo(model, mel_spec, le)
    if tempo is None:
        return jsonify({'error': 'Could not estimate tempo'}), 500

    return jsonify({'tempo': int(tempo)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)