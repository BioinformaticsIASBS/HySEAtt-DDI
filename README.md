# HySEAtt-DDI
# HySEAtt-DDI: Autoencoder–CNN–SE–Attention Pipeline for Drug–Drug Interaction Prediction

This repository contains the TensorFlow/Keras training pipeline used to evaluate an autoencoder-based hybrid neural network for five-class drug–drug interaction (DDI) prediction.

The implementation reads a drug-level similarity or feature matrix, reduces each drug representation with an autoencoder, builds pairwise drug features by concatenating the two latent vectors, and trains a hybrid classifier comprising convolutional layers, a Squeeze-and-Excitation (SE) block, positional embeddings, multi-head self-attention, and a multilayer perceptron (MLP).

> **Important:** The current script uses the file name `tanimoto_sim.txt` as its drug-level input matrix. If your study uses fused structural, biological, and semantic similarities, that fusion must be completed before running this script, or the loading section must be adapted to build the fused matrix inside the pipeline.

## Model overview

The pipeline follows these stages:

1. Load drug-level features from a text matrix.
2. Standardize features using `StandardScaler`.
3. Train an autoencoder to compress each drug vector into a latent representation of dimension 1,000.
4. Construct drug-pair features by concatenating the latent vectors of drug `i` and drug `j`.
5. Train a hybrid classifier with two 1D convolutional layers, batch normalization, max pooling, dropout, an SE block, positional embeddings, four-head self-attention, and an MLP classifier.
6. Evaluate with stratified 10-fold cross-validation and report micro, macro, and weighted metrics.

## Repository structure

```text
.
├── train_hyseatt.py
├── filtered_labels.txt
├── filtered_indices.txt
├── tanimoto_sim.txt
├── requirements.txt
└── README.md
```

After execution, results are saved in:

```text
AE_CNN_Att_MLP_Hybrid_results_1000_sequential/
└── Hybrid_Attention/
    ├── average_metrics.txt
    ├── fold_1/
    │   ├── metrics_Micro.txt
    │   ├── metrics_Macro.txt
    │   ├── metrics_Weighted.txt
    │   └── model.h5
    ├── ...
    └── fold_10/
```

## Requirements

* Python 3.9 or newer
* TensorFlow / Keras
* NumPy
* scikit-learn
* tqdm

Install dependencies:

```bash
pip install tensorflow numpy scikit-learn tqdm
```

## Input files

### `tanimoto_sim.txt`

A whitespace-delimited numerical matrix containing one feature vector per drug.

* Each row corresponds to a drug.
* The current implementation requires exactly **2,148 values per row**.
* Values are loaded as `float32`.
* Although the file is named `tanimoto_sim.txt`, it can contain another precomputed drug-level feature matrix with 2,148 columns.

### `filtered_indices.txt`

A whitespace-delimited list of drug pairs. Each line contains two zero-based drug indices:

```text
12 845
12 1630
845 1630
```

### `filtered_labels.txt`

One integer class label per drug pair, in the same order as `filtered_indices.txt`.

```text
0
3
1
```

The current model is configured for five classes:

```text
0, 1, 2, 3, 4
```

## Configuration

```python
num_classes = 5
num_folds = 10
batch_size = 64
epochs_mlp = 50
epochs_autoencoder = 10
encoding_dim = 1000
patience = 5
```

The classifier uses Adam with a learning rate of `0.0001`. The autoencoder uses Adam with mean squared error reconstruction loss.

## Running the pipeline

Save the supplied code as `train_hyseatt.py`, then run:

```bash
python train_hyseatt.py
```

The script:

1. Loads labels, drug pairs, and the drug-level matrix.
2. Standardizes the selected drug vectors.
3. Trains an autoencoder to create 1,000-dimensional latent features.
4. Concatenates the two latent drug vectors for each DDI pair.
5. Computes class-balanced weights.
6. Performs stratified 10-fold cross-validation.
7. Saves a trained model and metric files for every fold.
8. Writes mean ± standard deviation metrics to `average_metrics.txt`.

## Reported metrics

For each fold, the code reports:

* Accuracy
* Matthews correlation coefficient
* Precision
* Recall
* F1-score
* AUROC
* AUPR

Metrics are summarized using micro, macro, and weighted averaging.

## Reproducibility note

The current script uses pair-level stratified cross-validation:

```python
StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

It standardizes drug features and trains the autoencoder before the pair-level split. Therefore, it represents a transductive or known-drug evaluation setting: the same drug may occur in both training and test pairs.

For a strict unseen-drug evaluation, use a drug-disjoint split and fit the scaler and autoencoder using only the training drugs within each fold. Also ensure that reversed pairs such as `(i, j)` and `(j, i)` never appear in different partitions.

## Data and licensing

Do not upload raw DrugBank data or derivative records unless your DrugBank licence explicitly allows redistribution. The repository should contain code, data-processing scripts, permitted identifiers, and reproducible split files.

## Citation

```bibtex
@article{nouroozi2026hyseatt,
  title   = {HySEAtt-DDI: A multimodal deep learning framework for accurate drug--drug interaction prediction},
  author  = {Nouroozi, Mahdi and Hooshmand, Mohsen and Nasiri, Fatemeh},
  journal = {Journal of Cheminformatics},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Update the DOI, volume, issue, and pages after publication.

## Licence

Add an explicit open-source licence before making the repository public. MIT or Apache-2.0 are common choices for academic code repositories.

## Contact

For questions about the implementation or the associated manuscript, contact the corresponding author listed in the paper.
