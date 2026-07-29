# Heart Disease Prediction & CI/CD Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-green.svg)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange.svg)

## 📌 Deskripsi Proyek
Proyek ini dikembangkan oleh **Jauhari Achmad Pradana** untuk membangun model Machine Learning klasifikasi penyakit jantung (*Heart Disease Prediction*) menggunakan algoritma **Random Forest Classifier**. Pelatihan model ini diintegrasikan secara otomatis dengan **MLflow** untuk *experiment tracking* serta **GitHub Actions** untuk *Continuous Integration (CI)* pipeline.

---

## 📁 Struktur Direktori
```text
.
├── .github/
│   └── workflows/
│       └── ml-ci.yml          # Konfigurasi GitHub Actions CI Pipeline
├── MLProject/
│   ├── heart_disease_preprocessing/
│   │   ├── X_train.csv        # Feature data latih
│   │   ├── X_test.csv         # Feature data uji
│   │   ├── y_train.csv        # Target data latih
│   │   └── y_test.csv         # Target data uji
│   ├── conda.yaml             # Spesifikasi environment Conda & dependensi
│   ├── MLProject              # Definisi MLflow Project
│   └── modelling.py           # Script utama pelatihan model & autolog MLflow
├── .gitignore                 # Daftar file/folder yang diabaikan oleh Git
└── README.md                  # Dokumentasi utama proyek
```

---

## 🛠️ Instalasi & Eksekusi Lokal

### 1. Prasyarat
Pastikan Anda telah menginstal **Python 3.9+** atau lingkungan **Conda**.

### 2. Install Dependensi
Masuk ke direktori `MLProject` dan install paket yang dibutuhkan:
```bash
pip install mlflow==2.19.0 scikit-learn pandas numpy matplotlib
```

### 3. Jalankan Script Pelatihan Model
Jalankan script `modelling.py` secara langsung:
```bash
cd MLProject
python modelling.py
```

Atau menggunakan perintah **MLflow CLI**:
```bash
mlflow run MLProject
```

---

## 📊 Hasil Evaluasi Model
Model Random Forest dievaluasi menggunakan metrik standar klasifikasi:
- **Akurasi (Accuracy)**
- **Presisi (Precision)**
- **Daya Ingat (Recall)**
- **F1-Score**

Semua metrik dan parameter model secara otomatis dicatat ke dalam direktori `mlruns/` oleh MLflow Autolog.

---

## 🚀 Continuous Integration (GitHub Actions)
Pipeline CI dikonfigurasi dalam `.github/workflows/ml-ci.yml`. Setiap kali ada perubahan (*push*) ke branch `main`, GitHub Actions akan:
1. Menyiapkan environment Python 3.9.
2. Menginstall dependensi proyek.
3. Melatih model machine learning.
4. Mengunggah artefak hasil eksperimen MLflow (`mlruns/`).

---
*Dikembangkan oleh Jauhari Achmad Pradana.*
