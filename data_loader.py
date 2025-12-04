# Loads patient data (train, test, validation), just keep amplification data

import pandas as pd
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).resolve().parent

# These files should be downloaded from original database (github wouldn't allow uploading of these files bc they're too big
CNV_FILE = BASE_DIR / "P1000_data_CNA_paper.csv"
RESPONSE_FILE = BASE_DIR / "response_paper.csv"
TRAIN_FILE = BASE_DIR / "training_set_0.csv"
VAL_FILE = BASE_DIR / "validation_set.csv"
TEST_FILE = BASE_DIR / "test_set.csv"


def load_response() -> pd.Series:
    
    df = pd.read_csv(RESPONSE_FILE)
    # Expecting columns: ['id', 'response']
    df = df.set_index("id")
    # Ensure it's int (0/1)
    y = df["response"].astype(int)
    return y


def load_cnv_amp() -> pd.DataFrame:
    """
    Note: the P1000_data_CNA_paper.csv file has been processed by original authors already
        rows: tumor/sample IDs
        columns: gene symbols
        entries: -2, -1, 0, 1, 2

    """
    cnv = pd.read_csv(CNV_FILE, index_col=0)
    cnv_amp = cnv.copy()

    # Everything <= 0 is not an amplification
    cnv_amp[cnv_amp <= 0] = 0

    # Strong amplifications only
    cnv_amp[cnv_amp == 1] = 0
    cnv_amp[cnv_amp == 2] = 1

    cnv_amp = cnv_amp.astype(int)
    return cnv_amp


def load_splits() -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Load train/val/test splits and return three Index objects
    with the sample IDs in each split.
    """
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_ids = pd.Index(train_df["id"].astype(str))
    val_ids = pd.Index(val_df["id"].astype(str))
    test_ids = pd.Index(test_df["id"].astype(str))

    return train_ids, val_ids, test_ids


class ProstateMDM4Dataset:
    """
    """

    def __init__(self):
        # 1) Load everything
        self.cnv_amp = load_cnv_amp()          # DataFrame [samples x genes]
        self.y_all = load_response()           # Series   [samples]

        train_ids, val_ids, test_ids = load_splits()

        # 2) Restrict to samples that exist in both CNV and response
        common_ids = self.cnv_amp.index.intersection(self.y_all.index)
        self.cnv_amp = self.cnv_amp.loc[common_ids]
        self.y_all = self.y_all.loc[common_ids]

        # 3) Intersect splits with common_ids
        self.train_ids = train_ids.intersection(common_ids)
        self.val_ids = val_ids.intersection(common_ids)
        self.test_ids = test_ids.intersection(common_ids)

        # 4) Build matrices for each split
        self.X_train = self.cnv_amp.loc[self.train_ids].values
        self.y_train = self.y_all.loc[self.train_ids].values

        self.X_val = self.cnv_amp.loc[self.val_ids].values
        self.y_val = self.y_all.loc[self.val_ids].values

        self.X_test = self.cnv_amp.loc[self.test_ids].values
        self.y_test = self.y_all.loc[self.test_ids].values

        # Keep also the gene list and full matrix if you need it for masks
        self.genes = list(self.cnv_amp.columns)
        self.samples_all = list(self.cnv_amp.index)

    def get_train_val_test(self):
        return (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            self.genes,
        )

    def get_full_matrix(self):
        """
        Returns the full amplification matrix and labels for all samples.
        """
        return self.cnv_amp.copy(), self.y_all.copy()

from pathlib import Path


if __name__ == "__main__":
    dataset = ProstateMDM4Dataset()
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        genes,
    ) = dataset.get_train_val_test()

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:", pd.Series(y_train).value_counts(normalize=True))
    print("y_val distribution:", pd.Series(y_val).value_counts(normalize=True))
    print("y_test distribution:", pd.Series(y_test).value_counts(normalize=True))
    print("Number of genes (CNV_amp features):", len(genes))
    print("Example genes:", genes[:10])
