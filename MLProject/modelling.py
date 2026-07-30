"""
===================================================================
Project: Heart Disease Machine Learning Model & CI/CD Pipeline
Author: Jauhari Achmad Pradana
Description: Script pelatihan model klasifikasi penyakit jantung 
             menggunakan Random Forest Classifier dan terintegrasi 
             dengan MLflow Autologging.
===================================================================
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_dataset(data_directory: str = "heart_disease_preprocessing") -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Membaca data pelatihan dan pengujian yang telah dipreproses dari direktori data.
    """
    x_train_path = os.path.join(data_directory, "X_train.csv")
    x_test_path = os.path.join(data_directory, "X_test.csv")
    y_train_path = os.path.join(data_directory, "y_train.csv")
    y_test_path = os.path.join(data_directory, "y_test.csv")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    return X_train, X_test, y_train, y_test


def _train_and_evaluate(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> None:
    # Inisialisasi dan pelatihan model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = clf.predict(X_test)

    # Kalkulasi metrik evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Menampilkan ringkasan hasil evaluasi
    print("\n" + "=" * 40)
    print("     METRIK EVALUASI MODEL MLFLOW     ")
    print("=" * 40)
    print(f"  Akurasi (Accuracy)   : {acc:.4f}")
    print(f"  Presisi (Precision)  : {prec:.4f}")
    print(f"  Daya Ingat (Recall)  : {rec:.4f}")
    print(f"  F1 Score             : {f1:.4f}")
    print("=" * 40)
    print("[SUCCESS] Pelatihan model dan CI pipeline selesai dijalankan!\n")


def execute_training_pipeline() -> None:
    """
    Eksekusi alur pelatihan model Random Forest beserta pelacakan MLflow.
    """
    print("[INFO] Memuat dataset terpreproses...")
    X_train, X_test, y_train, y_test = load_dataset()

    print(f"[DATASET] Ukuran Data Latih  (X_train): {X_train.shape}")
    print(f"[DATASET] Ukuran Data Uji    (X_test) : {X_test.shape}")

    # Mengaktifkan pelacakan otomatis dari MLflow sklearn
    mlflow.sklearn.autolog()

    # Jika dipanggil lewat `mlflow run`, MLFLOW_RUN_ID diset di os.environ
    if "MLFLOW_RUN_ID" in os.environ:
        print(f"[INFO] Dijalankan via 'mlflow run' dengan MLFLOW_RUN_ID={os.environ['MLFLOW_RUN_ID']}")
        _train_and_evaluate(X_train, X_test, y_train, y_test)
    else:
        print("[INFO] Dijalankan secara langsung via Python script...")
        mlflow.set_experiment("Heart_Disease_Prediction")
        with mlflow.start_run(run_name="RandomForest_CI"):
            _train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    execute_training_pipeline()

