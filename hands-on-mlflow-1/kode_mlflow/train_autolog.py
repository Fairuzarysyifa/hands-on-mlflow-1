# train_autolog.py
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Aktifkan autologging untuk scikit-learn
mlflow.sklearn.autolog()

# 2. Hubungkan ke server MLflow lokal dan buat eksperimen baru
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-autolog")

# 3. Siapkan data
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Latih model (cukup panggil fit() saja, MLflow yang urus sisanya!)
with mlflow.start_run(run_name="rf-autolog-run"):
  model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
  model.fit(X_tr, y_tr)

print("Training dengan autologging selesai!")