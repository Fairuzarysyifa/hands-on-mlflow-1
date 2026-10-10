# train_optuna.py
import mlflow
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Siapkan data
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hubungkan ke MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("breast-cancer-optuna")


# Definisikan fungsi objektif untuk Optuna
def objective(trial):
  # Tentukan rentang parameter yang akan dicari otomatis
  params = {
      "n_estimators": trial.suggest_int("n_estimators", 50, 300),
      "max_depth": trial.suggest_int("max_depth", 3, 15),
      "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
      "random_state": 42,
  }

  # Buat nested run untuk setiap percobaan (trial) Optuna
  with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
    mlflow.log_params(params)

    # Latih model
    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)

    # Evaluasi
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)

    # Log metrik
    mlflow.log_metric("val_f1", f1)

  return f1


# Jalankan parent run untuk Optuna study
with mlflow.start_run(run_name="optuna-hyperparameter-tuning"):
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=5)  # Kita coba 5 kali percobaan

  # Log parameter terbaik ke parent run
  mlflow.log_params(study.best_params)
  mlflow.log_metric("best_val_f1", study.best_value)

print("Hyperparameter tuning dengan Optuna selesai!")