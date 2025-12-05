# MDM4-Anchored Biologically Informed Neural Network

**CS 679 Final Project — Morghan van Walsum & Saranya Varakunan**

This repository contains all code used to implement and evaluate an MDM4-centric biologically informed neural network inspired by the P-NET architecture introduced in Elmarakeby et al. (2021). The goal of this project is to determine whether pathways directly involving the gene MDM4 contain sufficient biological signal to predict metastatic progression in prostate cancer. For comparison, we also include a fully connected baseline model with a matched number of parameters.
The dataset used in this project is the P1000 prostate cancer cohort (Armenia et al, 2019), which includes gene-level copy-number variation (CNV) profiles for approximately 13,800 genes and clinical labels indicating whether each sample is from a primary or metastatic tumour. The dataset is not included in this repository because the files exceed GitHub size limits; you must download them manually from the original database.

**Repository Structure**

The file data_loader.py handles all data ingestion and preprocessing. It loads the CNV matrix, clinical responses, and training/validation/test splits. CNV values are transformed into binary amplification features to match the modelling choices described in the original paper. This file ensures that gene ordering is consistent across all components of the pipeline.

The file pathways.py loads KEGG and Reactome pathway collections from provided GMT files. It identifies all pathways that contain the gene MDM4 and constructs binary mask matrices mapping genes to pathways. These masks define the sparse, biologically informed connectivity used in the MDM4-anchored model.

The file model_mdm4.py defines the PyTorch implementation of the MDM4-centric pathway network. It includes masked linear layers representing MDM4-related KEGG and Reactome pathways, followed by a small fully connected layer and an output neuron producing a binary classification logit. This structure mirrors the layered biological hierarchy used in P-NET while restricting the model strictly to MDM4-associated pathways.

The file model_baseline.py implements a fully connected multilayer perceptron (MLP) with three hidden layers (25, 20, and 12 neurons). Its parameter count is matched to the MDM4 model (approximately 3.46 × 10⁵ parameters) to enable a fair architectural comparison. Unlike the MDM4 model, it contains no biological constraints.

**The notebook run_pnet_mdm4.ipynb** (and the script run_pnet_mdm4.py) is the main training and evaluation script for the biologically informed model. It builds a biologically informed model based on MDM4 pathways extracted from .gmt files and prints the identified pathways, trains the model (data needs to be downloaded separately), and prints training time and test AUC.

The script run_model_baseline.py trains the fully connected baseline model using the same training pipeline (batching, L2 regularization, early stopping, Adam optimizer). 

The GMT pathway files (kegg.gmt and reactome.gmt) are included in the repository and are used to determine MDM4 pathway membership.

**Required External Data Files**

The following files must be downloaded from the database. They come from the original P-NET dataset and cannot be uploaded here due to size limitations:

P1000_data_CNA_paper.csv        # CNV matrix (~13,802 genes × patients)
response_paper.csv              # Labels: 0 = Primary, 1 = Metastatic

The paths to these files are defined inside data_loader.py:
CNV_FILE = BASE_DIR / "P1000_data_CNA_paper.csv"
RESPONSE_FILE = BASE_DIR / "response_paper.csv"

Once these files are present, models can be trained and evaluated.

**How to Run the Models**

To train and test the MDM4-anchored biologically informed model:
python run_pnet_mdm4.py

To train and test the fully connected baseline:
python run_model_baseline.py

Both scripts should automatically report validation and test AUC, accuracy, loss curves, and runtime.

**Summary**

This project demonstrates that a biologically constrained architecture can achieve strong predictive performance while being more interpretable than an equally sized fully connected model. The results support the significance of MDM4-related pathways in metastatic prostate cancer biology. The repository provides a clean and modular implementation that can be extended to other genes, pathways, or disease contexts.
