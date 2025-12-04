# Based on original paper's code. Will construct masks for genes to MDM4 pathways
# Can alter the "MDM4" to any other gene (e.g. our ACTB negative control) to make a mask for that gene's related pathways!!

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def load_gmt_generic(
    gmt_path: Path,
    pathway_col: int,
    genes_start_col: int,
    filter_genes: Optional[List[str]] = None,
    min_genes: int = 1,
) -> Dict[str, List[str]]:
    """
    Load a GMT file and return pathways as:
        dict[pathway_name -> list of genes]
    """
    pathways: Dict[str, List[str]] = {}

    with gmt_path.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= genes_start_col:
                continue

            pathway_name = parts[pathway_col]
            genes = parts[genes_start_col:]

            if filter_genes is not None:
                gene_set = set(filter_genes)
                genes = [g for g in genes if g in gene_set]

            if len(genes) >= min_genes:
                pathways[pathway_name] = genes

    return pathways


def build_gene_pathway_matrix(
    pathways: Dict[str, List[str]],
    all_genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a gene x pathway binary matrix (DataFrame) from:
        dict[pathway_name -> list of genes]
    """
    gene_set = set()
    for genes in pathways.values():
        gene_set.update(genes)

    if all_genes is None:
        genes = sorted(gene_set)
    else:
        genes = [g for g in all_genes if g in gene_set]

    df = pd.DataFrame(
        0,
        index=genes,
        columns=list(pathways.keys()),
        dtype=int,
    )

    for pathway, gene_list in pathways.items():
        for g in gene_list:
            if g in df.index:
                df.at[g, pathway] = 1

    return df


def extract_mdm4_submatrix(gene_pathway_df: pd.DataFrame) -> pd.DataFrame:
    """
    From a gene x pathway matrix, keep only pathways that contain MDM4.
    Returns a matrix genes x (MDM4 pathways). May be empty if none.
    """
    if "MDM4" not in gene_pathway_df.index:
        print("MDM4 is not present in this matrix.")
        return pd.DataFrame(index=gene_pathway_df.index)

    mdm4_pathways = [
        p for p in gene_pathway_df.columns
        if gene_pathway_df.loc["MDM4", p] == 1
    ]

    print(f"MDM4 is present in {len(mdm4_pathways)} pathways.")
    if not mdm4_pathways:
        return pd.DataFrame(index=gene_pathway_df.index)

    print("MDM4 pathways:", mdm4_pathways)
    mdm4_df = gene_pathway_df[mdm4_pathways]
    print("MDM4-centric matrix shape (genes x mdm4_pathways):", mdm4_df.shape)
    return mdm4_df


def build_mask_for_gene_order(
    gene_pathway_df: pd.DataFrame,
    gene_order: List[str],
) -> pd.DataFrame:
    """
    Reorder rows to match a given gene_order, filling missing genes with zeros. Returns a gene x pathway mask.
    """
    mask_df = gene_pathway_df.reindex(index=gene_order).fillna(0).astype(int)
    return mask_df

from pathlib import Path

def get_mdm4_masks_for_gene_order(gene_order):
    """
    Given a list of genes (e.g. columns of X from data_loader),
    return KEGG + Reactome MDM4 pathway masks aligned to that order.

    """
    base_dir = Path(__file__).resolve().parent

    # ---------- KEGG ----------
    kegg_file = base_dir / "c2.cp.kegg.v6.1.symbols.gmt"
    kegg_pathways = load_gmt_generic(
        gmt_path=kegg_file,
        pathway_col=0,
        genes_start_col=2,
        filter_genes=None,
        min_genes=3,
    )
    kegg_gene_pathway_df = build_gene_pathway_matrix(kegg_pathways)
    kegg_mdm4_df = extract_mdm4_submatrix(kegg_gene_pathway_df)

    # Align to gene_order (CNV genes)
    kegg_mask_df = build_mask_for_gene_order(kegg_mdm4_df, gene_order)

    # ---------- Reactome ----------
    reactome_file = base_dir / "ReactomePathways.gmt"
    reactome_pathways = load_gmt_generic(
        gmt_path=reactome_file,
        pathway_col=0,      # human-readable name
        genes_start_col=2,  # genes start at column 2
        filter_genes=None,
        min_genes=3,
    )
    reactome_gene_pathway_df = build_gene_pathway_matrix(reactome_pathways)
    reactome_mdm4_df = extract_mdm4_submatrix(reactome_gene_pathway_df)

    # Align to gene_order
    reactome_mask_df = build_mask_for_gene_order(reactome_mdm4_df, gene_order)

    return kegg_mask_df, reactome_mask_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # =========================
    # KEGG (MSigDB C2 KEGG GMT)
    # =========================
    
    # Note on format:
    #   col0: pathway name
    #   col1: URL / description
    #   col2+: gene symbols
    
    kegg_file = base_dir / "c2.cp.kegg.v6.1.symbols.gmt"

    kegg_pathways = load_gmt_generic(
        gmt_path=kegg_file,
        pathway_col=0,
        genes_start_col=2,
        filter_genes=None,
        min_genes=3,
    )

    kegg_gene_pathway_df = build_gene_pathway_matrix(kegg_pathways)

    print("=== KEGG ===")
    print("Number of KEGG pathways:", len(kegg_pathways))
    print("KEGG matrix shape (genes x pathways):", kegg_gene_pathway_df.shape)
    print(kegg_gene_pathway_df.head())

    kegg_mdm4_df = extract_mdm4_submatrix(kegg_gene_pathway_df)

    # =========================
    # Reactome GMT 
    # =========================
    
    reactome_file = base_dir / "ReactomePathways.gmt"

    reactome_pathways = load_gmt_generic(
        gmt_path=reactome_file,
        pathway_col=0,      # use the actual pathway name 
        genes_start_col=2,  # genes start at column 2
        filter_genes=None,
        min_genes=3,
    )

    reactome_gene_pathway_df = build_gene_pathway_matrix(reactome_pathways)

    print("\n=== Reactome ===")
    print("Number of Reactome pathways:", len(reactome_pathways))
    print("Reactome matrix shape (genes x pathways):", reactome_gene_pathway_df.shape)
    print(reactome_gene_pathway_df.head())

    reactome_mdm4_df = extract_mdm4_submatrix(reactome_gene_pathway_df)

    # So now we have 
    #   kegg_mdm4_df     := genes x [KEGG pathways containing MDM4]
    #   reactome_mdm4_df := genes x [Reactome pathways containing MDM4]
  
