
# Makes a simple MDM4-centric pathway network.
# - Input:  gene-level CNV features (binary amplifications-only)
# - Hidden: KEGG + Reactome MDM4 pathway layers 
# - Output: binary classification: Primary vs Metastasis

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """
    Linear layer whose weights are multiplied elementwise by a fixed mask. (A simpler analogue of the sparse connection layers in the original paper)

    """

    def __init__(self, in_features: int, out_features: int, mask, bias: bool = True):
        super().__init__()

        # Convert mask to torch tensor and check shape
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
        assert mask_tensor.shape == (out_features, in_features), (
            f"Mask shape {mask_tensor.shape} does not match "
            f"(out_features={out_features}, in_features={in_features})"
        )

        self.in_features = in_features
        self.out_features = out_features

        # Trainable weights 
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Mask (note: DO NOT TRAIN!!!!!!!)
        self.register_buffer("mask", mask_tensor)

        # Initialization for weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_features)
        returns: (batch_size, out_features)
        """
        # Apply mask to weights before linear transform
        effective_weight = self.weight * self.mask  # (out_features, in_features)
        return F.linear(x, effective_weight, self.bias)


class MDM4PathwayNet(nn.Module):
    """
    MDM4 pathway model:
    genes (CNV amps) → KEGG-MDM4 pathway layer → Reactome-MDM4 pathway layer → dense hidden → output logit

    n_genes : int
        Number of gene features (columns of X_*).
    kegg_mask : gene i belongs to KEGG pathway j if mask[i, j] = 1.
    reactome_mask : Same as KEGG.
    hidden_dim : size of the dense hidden layer on top of concatenated pathways.
    dropout : sropout probability before the dense hidden layer.

    """

    def __init__(
        self,
        n_genes: int,
        kegg_mask,
        reactome_mask,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Figure out how many pathways we have from the masks
        n_kegg = kegg_mask.shape[1]
        n_reactome = reactome_mask.shape[1]

        # Gene → KEGG MDM4 pathways
        # Transpose the gene×pathway masks to pathway×gene, bc the maskedlinear expects (out_features, in_features)
        self.kegg_layer = MaskedLinear(
            in_features=n_genes,
            out_features=n_kegg,
            mask=kegg_mask.T,  # (n_kegg, n_genes)
            bias=True,
        )

        # Gene → Reactome MDM4 pathways
        self.reactome_layer = MaskedLinear(
            in_features=n_genes,
            out_features=n_reactome,
            mask=reactome_mask.T,  # (n_reactome, n_genes)
            bias=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Dense MLP head on concatenated pathway activations
        self.fc = nn.Linear(n_kegg + n_reactome, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  # binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_genes)  – same gene order as used to build the masks.
        returns: (batch_size,) logits
        """
        # Pathway activations (ReLU like in original P-NET-style models)
        k = torch.relu(self.kegg_layer(x))       # (batch, n_kegg)
        r = torch.relu(self.reactome_layer(x))   # (batch, n_reactome)

        # Concatenate 
        h = torch.cat([k, r], dim=1)             # (batch, n_kegg + n_reactome)
        h = self.dropout(h)
        h = torch.relu(self.fc(h))               # (batch, hidden_dim)

        logit = self.out(h)                      # (batch, 1)
        return logit.squeeze(1)                  # (batch,)
