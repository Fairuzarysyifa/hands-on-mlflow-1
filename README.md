# Praktikum MLOps
 
Repositori ini berisi pengerjaan tugas praktikum MLOps. Setiap hands-on disimpan dalam folder terpisah agar kode, hasil eksperimen, dan dokumentasinya tertata.
 
**Nama:** Fairuz Ary Syifa · **NIM:** 123450044 · **Kelas:** RA
 
## Hands-on 1 — Pelacakan Eksperimen dengan MLflow
 
Folder [`Hands-on-1/`](Hands-on-1/) berisi implementasi pelacakan eksperimen klasifikasi kanker payudara (*Breast Cancer Wisconsin*) menggunakan MLflow, scikit-learn, PyTorch, dan Optuna. Di dalamnya terdapat kode untuk logging dasar, logging per epoch, autologging, hyperparameter tuning dengan nested runs, pencarian run secara programatik, serta pendaftaran model ke Model Registry.
 
### Cara singkat menjalankan
 
```bash
python -m venv .venv
.venv\Scripts\activate
pip install mlflow scikit-learn pandas torch torchvision optuna
 
# Terminal 1: jalankan server MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
 
# Terminal 2: jalankan skrip berurutan
python train_basic.py
python train_pytorch.py
python train_autolog.py
python train_optuna.py
python search_runs.py
python train_and_register.py
```
 
MLflow UI dapat dibuka di <http://localhost:5000>.
 
