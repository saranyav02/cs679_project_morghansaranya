from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import ProstateMDM4Dataset
from model_baseline import BaselineMLP


# --------------------------
# 1) Load data
# --------------------------
dataset = ProstateMDM4Dataset()
(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    genes
) = dataset.get_train_val_test()

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:  ", X_val.shape, y_val.shape)
print("Test shape: ", X_test.shape, y_test.shape)

X_train_t = torch.from_numpy(X_train.astype("float32"))
X_val_t   = torch.from_numpy(X_val.astype("float32"))
X_test_t  = torch.from_numpy(X_test.astype("float32"))

y_train_t = torch.from_numpy(y_train.astype("float32"))
y_val_t   = torch.from_numpy(y_val.astype("float32"))
y_test_t  = torch.from_numpy(y_test.astype("float32"))

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=64)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=64)

# --------------------------
# 2) Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_dim = X_train.shape[1]      # number of CNV_amp features (genes)
model = BaselineMLP(input_dim=input_dim, dropout=0.3).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)


# --------------------------
# 3) Training helpers (based on original code)
# --------------------------
def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_targets = []
    all_probs = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(xb)
        # ensure shape (batch,)
        logits = logits.view(-1)

        loss = criterion(logits, yb)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)

        with torch.no_grad():
            probs_batch = torch.sigmoid(logits)

        all_targets.append(yb.cpu().numpy())
        all_probs.append(probs_batch.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_targets, preds)
    try:
        auc = roc_auc_score(all_targets, probs)
    except ValueError:
        auc = np.nan

    return total_loss / len(loader.dataset), acc, auc


# --------------------------
# 4) Train with early stopping
# --------------------------
best_auc = -np.inf
best_state = None
patience = 15
patience_counter = 0

for epoch in range(1, 200):
    tr_loss, tr_acc, tr_auc = run_epoch(train_loader, train=True)
    va_loss, va_acc, va_auc = run_epoch(val_loader,   train=False)

    print(f"Epoch {epoch:03d} | Train AUC {tr_auc:.3f} | Val AUC {va_auc:.3f}")

    if va_auc > best_auc:
        best_auc = va_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best AUC = {best_auc:.3f}")
            break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

# --------------------------
# 5) Test evaluation
# --------------------------
te_loss, te_acc, te_auc = run_epoch(test_loader, train=False)
print("\nBaseline MLP Test Performance:")
print(f"  Loss: {te_loss:.4f}")
print(f"  Acc : {te_acc:.3f}")
print(f"  AUC : {te_auc:.3f}")
