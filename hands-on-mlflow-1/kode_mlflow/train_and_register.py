# train_and_register.py
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Hubungkan ke server MLflow lokal
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-optuna")

# 2. Siapkan data
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Mulai run baru untuk melatih dan mendaftarkan model secara langsung
with mlflow.start_run(run_name="model-registry-run") as run:
  params = {"n_estimators": 219, "max_depth": 3, "min_samples_split": 3}
  mlflow.log_params(params)

  # Latih model
  model = RandomForestClassifier(**params, random_state=42)
  model.fit(X_tr, y_tr)

  # Log model menggunakan serialization_format="cloudpickle" agar lolos dari validasi skops
  mlflow.sklearn.log_model(
      model, name="model", serialization_format="cloudpickle"
  )

  # 4. Daftarkan model ke Model Registry
  model_uri = f"runs:/{run.info.run_id}/model"
  name = "breast-cancer-clf"

  print("Melakukan registrasi model...")
  mv = mlflow.register_model(model_uri, name)
  print(f"Model berhasil didaftarkan! Versi model: {mv.version}")

  # 5. Berikan anotasi, tag, dan alias (@challenger & @champion)
  client = mlflow.MlflowClient()
  client.update_model_version(
      name, mv.version, description="RF depth=3, hasil tuning Optuna"
  )
  client.set_model_version_tag(name, mv.version, "validation_status", "passed")
  client.set_registered_model_alias(name, "challenger", mv.version)
  client.set_registered_model_alias(name, "champion", mv.version)
  print("Alias '@challenger' dan '@champion' berhasil disematkan!")

  # 6. Uji muat model via alias
  loaded_model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
  print("Model berhasil dimuat ulang menggunakan alias '@champion'!")