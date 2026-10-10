# train_pytorch.py
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim


# Fungsi dummy untuk simulasi training per epoch (sesuai alur modul)
def train_one_epoch(model, dataloader, criterion, optimizer):
  model.train()
  # Simulasi nilai training loss
  return 0.25


def evaluate(model, dataloader, criterion):
  model.eval()
  # Simulasi nilai validation loss dan F1 score
  return 0.15, 0.88


# Konfigurasi parameter (cfg)
cfg = {
    "lr": 3e-4,
    "batch_size": 64,
    "epochs": 5,  # Dibuat 5 epoch agar cepat sebagai uji coba
    "arch": "simple_nn",
    "seed": 42,
}

torch.manual_seed(cfg["seed"])

# Hubungkan ke server MLflow lokal
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("pytorch-epoch-logging")

# Definisikan model sederhana, loss, dan optimizer
model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

# Mulai MLflow Run
with mlflow.start_run(run_name=f"{cfg['arch']}-lr{cfg['lr']}"):
  # 1. Log parameter dan artifact
  mlflow.log_params(cfg)
  mlflow.log_artifact(__file__)  # Mencatat file skrip ini sebagai artifact

  best_f1 = 0.0

  # 2. Looping per epoch sesuai modul slide 44
  for epoch in range(cfg["epochs"]):
    # Simulasi data loader (dummy data)
    train_loss = train_one_epoch(model, None, criterion, optimizer)
    val_loss, val_f1 = evaluate(model, None, criterion)

    # Log metrics per epoch
    mlflow.log_metrics(
        {"train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1},
        step=epoch,
    )

    # Simpan checkpoint terbaik
    if val_f1 > best_f1:
      best_f1 = val_f1
      torch.save(model.state_dict(), "best.pt")

  # Log checkpoint model terbaik ke artifact MLflow
  mlflow.log_artifact("best.pt", artifact_path="checkpoints")

  # Log metrik akhir dan model PyTorch
  mlflow.log_metric("best_val_f1", best_f1)
  # Ubah bagian log_model di baris paling bawah menjadi:
input_example = torch.randn(1, 10)
mlflow.pytorch.log_model(model, name="model", input_example=input_example)

print("Training PyTorch per epoch selesai dan berhasil dicatat ke MLflow!")