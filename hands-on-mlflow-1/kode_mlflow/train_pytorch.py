# train_pytorch.py
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim


# Fungsi dummy untuk simulasi training per epoch (sesuai alur modul)
def train_one_epoch(model, dataloader, criterion, optimizer):
  model.train()
  return 0.25


def evaluate(model, dataloader, criterion):
  model.eval()
  return 0.15, 0.88


# Konfigurasi parameter (cfg)
cfg = {
    "lr": 3e-4,
    "batch_size": 64,
    "epochs": 5,
    "arch": "resnet18",
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
with mlflow.start_run(run_name=f"{cfg['arch']}-lr{cfg['lr']}" ):
  mlflow.log_params(cfg)
  mlflow.log_artifact(__file__)

  best_f1 = 0.0

  # Looping per epoch sesuai modul slide 44-45
  for epoch in range(cfg["epochs"]):
    train_loss = train_one_epoch(model, None, criterion, optimizer)
    val_loss, val_f1 = evaluate(model, None, criterion)

    # Log metrics per epoch
    mlflow.log_metrics(
        {"train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1},
        step=epoch,
    )

    if val_f1 > best_f1:
      best_f1 = val_f1
      torch.save(model.state_dict(), "best.pt")

  mlflow.log_artifact("best.pt", artifact_path="checkpoints")
  mlflow.log_metric("best_val_f1", best_f1)

  # Log model PyTorch dengan input_example dan serialization format pickle
  input_example = torch.randn(1, 10)
  mlflow.pytorch.log_model(
      model,
      name="model",
      input_example=input_example,
      serialization_format="pickle",
  )

print("Training PyTorch per epoch selesai dan berhasil dicatat ke MLflow!")