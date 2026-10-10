# train_basic.py
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Hubungkan ke server MLflow lokal dan buat nama eksperimen
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-clf")

# Siapkan data
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definisikan parameter model
params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}

# Mulai proses pencatatan (Run) di MLflow
with mlflow.start_run(run_name="rf-baseline"):
  # Beri tag tambahan
  mlflow.set_tags({"stage": "baseline", "author": "pemula"})

  # Log parameter
  mlflow.log_params(params)

  # Latih model Random Forest
  model = RandomForestClassifier(**params).fit(X_tr, y_tr)
  proba = model.predict_proba(X_val)[:, 1]

  # Hitung dan log metrik
  mlflow.log_metric("val_f1", f1_score(y_val, proba > 0.5))
  mlflow.log_metric("val_auc", roc_auc_score(y_val, proba))

  # Log model ke MLflow
  # Ganti baris mlflow.sklearn.log_model dengan kode ini:
mlflow.sklearn.log_model(
    model,
    name="model",
    input_example=X_val[:5],
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)

print("Training selesai dan berhasil dicatat ke MLflow!")