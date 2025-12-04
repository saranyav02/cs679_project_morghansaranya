# Note on data: train the MDM4PathwayNet on
#   - CNV_amp features (P1000_data_CNA_paper.csv → binary amps)
#   - response_paper.csv (0 = Primary, 1 = Metastasis)
#   - training_set_0.csv / validation_set.csv / test_set.csv (from database)


from pathlib import Path
import time  
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from data_loader import ProstateMDM4Dataset
from pathways import get_mdm4_masks_for_gene_order
from model_mdm4 import MDM4PathwayNet




# -------------------------
# 1) Load data
# -------------------------

base_dir = Path(__file__).resolve().parent

dataset = ProstateMDM4Dataset()

(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    genes,        # list of gene names in the same order as the columns of X_*
) = dataset.get_train_val_test()

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:  ", X_val.shape, y_val.shape)
print("Test shape: ", X_test.shape, y_test.shape)


def summarize_labels(name, y):
    vals, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    props = {int(v): counts[i] / float(total) for i, v in enumerate(vals)}
    print(f"{name} label distribution:", props)


summarize_labels("Train", y_train)
summarize_labels("Val",   y_val)
summarize_labels("Test",  y_test)

print("Number of genes (CNV_amp features):", len(genes))
print("Example genes:", genes[:10])

# -------------------------
# 2) Build KEGG + Reactome MDM4 masks
# -------------------------

kegg_mask_df, reactome_mask_df = get_mdm4_masks_for_gene_order(genes)

print("KEGG MDM4 mask shape:     ", kegg_mask_df.shape)
print("Reactome MDM4 mask shape: ", reactome_mask_df.shape)

kegg_mask = kegg_mask_df.to_numpy(dtype="float32")          # (n_genes, n_kegg)
reactome_mask = reactome_mask_df.to_numpy(dtype="float32")  # (n_genes, n_reactome)

assert list(kegg_mask_df.index) == genes
assert list(reactome_mask_df.index) == genes

print("Done building data + MDM4 pathway masks.")

# -------------------------
# 3) Torch setup
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Convert numpy arrays to torch tensors
X_train_t = torch.from_numpy(X_train.astype("float32"))
X_val_t   = torch.from_numpy(X_val.astype("float32"))
X_test_t  = torch.from_numpy(X_test.astype("float32"))

# BCEWithLogitsLoss expects float targets 0.0 / 1.0
y_train_t = torch.from_numpy(y_train.astype("float32"))
y_val_t   = torch.from_numpy(y_val.astype("float32"))
y_test_t  = torch.from_numpy(y_test.astype("float32"))

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

batch_size = 64

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# -------------------------
# 4) Build model
# -------------------------

model = MDM4PathwayNet(
    n_genes=len(genes),
    kegg_mask=kegg_mask,
    reactome_mask=reactome_mask,
    hidden_dim=32,
    dropout=0.2,
).to(device)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-5,
)

# -------------------------
# 5) Training + evaluation helpers
# -------------------------

def run_epoch(loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_targets = []
    all_logits = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(xb)  # (batch,)
        loss = criterion(logits, yb)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_targets.append(yb.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    total_loss /= len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_logits = np.concatenate(all_logits)

    # Convert logits → probabilities
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_targets, preds)
    try:
        auc = roc_auc_score(all_targets, probs)
    except ValueError:
        auc = np.nan  # e.g. if only one class present in a split

    return total_loss, acc, auc


# -------------------------
# 6) Train loop 
# -------------------------

num_epochs = 200
best_val_auc = -np.inf
best_state = None

patience = 15
epochs_no_improve = 0

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc, train_auc = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_auc = run_epoch(val_loader, train=False)

    print(
        f"Epoch {epoch:03d} | "
        f"Train loss {train_loss:.4f}, acc {train_acc:.3f}, AUC {train_auc:.3f} | "
        f"Val loss {val_loss:.4f}, acc {val_acc:.3f}, AUC {val_auc:.3f}"
    )

    if val_auc > best_val_auc + 1e-4:  # tiny margin to avoid noise
        best_val_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
        print(f"  → New best val AUC: {best_val_auc:.4f}")
    else:
        epochs_no_improve += 1
        print(f"  → No improvement in val AUC for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best val AUC = {best_val_auc:.4f}"
            )
            break

# -------------------------
# 7) Test performance 
# -------------------------

if best_state is not None:
    model.load_state_dict(best_state)

test_loss, test_acc, test_auc = run_epoch(test_loader, train=False)
print("\nTest performance (best val AUC model):")
print(f"  Loss: {test_loss:.4f}")
print(f"  Acc : {test_acc:.3f}")
print(f"  AUC : {test_auc:.3f}")

