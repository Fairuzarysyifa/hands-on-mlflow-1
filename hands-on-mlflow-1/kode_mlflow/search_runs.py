# search_runs.py
import mlflow

# Melakukan pencarian run secara programatik berdasarkan filter dan urutan metrik
runs = mlflow.search_runs(
    experiment_names=["breast-cancer-optuna"],
    filter_string="metrics.val_f1 > 0.90",
    order_by=["metrics.val_f1 DESC"],
    max_results=10,
)

# Menampilkan kolom penting dari hasil pencarian runs
print(
    runs[["run_id", "params.n_estimators", "params.max_depth", "metrics.val_f1"]]
)

# Mengambil run_id terbaik dari baris pertama (index 0)
if not runs.empty:
  best_run_id = runs.iloc[0]["run_id"]
  print(f"\nRun ID terbaik ditemukan: {best_run_id}")
else:
  print("Tidak ada run yang sesuai dengan filter.")