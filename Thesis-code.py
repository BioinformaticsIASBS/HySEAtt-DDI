# Run this notebook with:
# jupyter nbconvert --to notebook --execute your_notebook.ipynb --output executed_notebook.ipynb > notebook_run.log 2>&1

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import (Input, Dense, Conv1D, BatchNormalization,
                                     MaxPooling1D, Flatten, Reshape, Dropout, MultiHeadAttention,
                                     GlobalAveragePooling1D, Multiply, Add, Lambda, Embedding, Layer)
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
                             roc_auc_score, average_precision_score)
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from tensorflow.keras import backend as K

# ===========================================
# Parameters
# ===========================================
num_classes = 5
num_folds = 10
batch_size = 64
epochs_mlp = 50
epochs_autoencoder = 10
encoding_dim = 1000
patience = 5




class PositionalEmbedding(Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.embedding(positions)  # (seq_len, embed_dim)
        pos_embeddings = tf.expand_dims(pos_embeddings, axis=0)  # (1, seq_len, embed_dim)
        return x + pos_embeddings

# ===========================================
# Data Loading Functions
# ===========================================
def load_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in tqdm(lines, desc=f"Loading {os.path.basename(filepath)}"):
        row = np.fromstring(line.strip(), sep=' ', dtype=np.float32)
        if len(row) != 2148:
            raise ValueError(f"Row has {len(row)} values (expected 2148)")
        data.append(row)
    return np.array(data)

def read_label_file(file_path):
    with open(file_path, 'r') as file:
        return np.array([int(line.strip()) for line in file])

def read_combinations_file(file_path):
    combinations = []
    with open(file_path, 'r') as file:
        for line in file:
            idx1, idx2 = map(int, line.strip().split())
            combinations.append((idx1, idx2))
    return combinations

# ===========================================
# Autoencoder
# ===========================================
def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(input_layer)
    x = Dense(256, activation='relu')(x)
    bottleneck = Dense(encoding_dim, activation='relu')(x)
    x = Dense(256, activation='relu')(bottleneck)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(input_dim, activation='linear')(x)
    autoencoder = KerasModel(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder_model = KerasModel(inputs=input_layer, outputs=bottleneck)
    return autoencoder, encoder_model

def reduce_features(X, encoding_dim):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    autoencoder, encoder_model = build_autoencoder(X.shape[1], encoding_dim)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs_autoencoder,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[TqdmCallback(verbose=1), early_stopping]
    )
    return encoder_model.predict(X_scaled, verbose=0)


# ===========================================
# Hybrid Model
# ===========================================

def se_block(input_tensor, reduction=8):
    filters = K.int_shape(input_tensor)[-1]   # ✅ returns integer, not None
    se = GlobalAveragePooling1D()(input_tensor)
    se = Dense(max(1, filters // reduction), activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, filters))(se)
    return Multiply()([input_tensor, se])

def build_hybrid_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = Reshape((-1, 40))(input_layer)   # reshape safely

    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = se_block(x)  # Squeeze-and-Excitation Block

    # After SE block
    x = PositionalEmbedding(max_len=100, embed_dim=32)(x)



    attn_output = MultiHeadAttention(num_heads=4, key_dim=16)(query=x, value=x, key=x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(input_layer, output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def compute_metrics(y_true, y_pred, y_probs, num_classes):
    metrics = {"Micro": {}, "Macro": {}, "Weighted": {}}

    # Handle NaNs in probabilities
    if np.isnan(y_probs).any():
        y_probs = np.nan_to_num(y_probs, nan=0.0)

    # Fix for binary case: expand to 2-column probs
    if y_probs.ndim == 1 or y_probs.shape[1] == 1:
        y_probs = np.vstack([1 - y_probs, y_probs.ravel()]).T

    # Micro metrics
    metrics["Micro"]["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Micro"]["MCC"] = matthews_corrcoef(y_true, y_pred)

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["Micro"].update({"Precision": p, "Recall": r, "F1": f})

    # Micro AUC
    try:
        metrics["Micro"]["AUC"] = roc_auc_score(
            y_true, y_probs, average="micro", multi_class="ovr"
        )
    except Exception:
        metrics["Micro"]["AUC"] = np.nan

    # Micro AUPR
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        metrics["Micro"]["AUPR"] = average_precision_score(
            y_true_bin, y_probs, average="micro"
        )
    except Exception:
        metrics["Micro"]["AUPR"] = np.nan

    # Per-class metrics
    try:
        ppc, rpc, fpc, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        aupr_pc = []
        for i in range(num_classes):
            try:
                aupr_pc.append(
                    average_precision_score((y_true == i).astype(int), y_probs[:, i])
                )
            except Exception:
                aupr_pc.append(np.nan)

        aupr_pc = np.array(aupr_pc)
        weights = np.bincount(y_true, minlength=num_classes) / len(y_true)

        # Macro
        metrics["Macro"]["Precision"] = np.nanmean(ppc)
        metrics["Macro"]["Recall"] = np.nanmean(rpc)
        metrics["Macro"]["F1"] = np.nanmean(fpc)
        metrics["Macro"]["AUPR"] = np.nanmean(aupr_pc)
        try:
            metrics["Macro"]["AUC"] = roc_auc_score(
                y_true, y_probs, average="macro", multi_class="ovr"
            )
        except Exception:
            metrics["Macro"]["AUC"] = np.nan

        # Weighted
        metrics["Weighted"]["Precision"] = np.nansum(ppc * weights)
        metrics["Weighted"]["Recall"] = np.nansum(rpc * weights)
        metrics["Weighted"]["F1"] = np.nansum(fpc * weights)
        metrics["Weighted"]["AUPR"] = np.nansum(aupr_pc * weights)
        try:
            metrics["Weighted"]["AUC"] = roc_auc_score(
                y_true, y_probs, average="weighted", multi_class="ovr"
            )
        except Exception:
            metrics["Weighted"]["AUC"] = np.nan

    except Exception:
        for avg in ["Macro", "Weighted"]:
            for k in ["Precision", "Recall", "F1", "AUPR", "AUC"]:
                metrics[avg][k] = np.nan

    return metrics

# ===========================================
# Main Pipeline
# ===========================================
def main():
    labels = read_label_file('filtered_labels.txt')
    combinations = read_combinations_file('filtered_indices.txt')
    drug_matrix = load_data('tanimoto_sim.txt')

    used_indices = sorted(set(i for pair in combinations for i in pair))
    filtered_matrix = drug_matrix[used_indices]
    reduced_partial = reduce_features(filtered_matrix, encoding_dim)

    reduced = np.zeros((drug_matrix.shape[0], encoding_dim), dtype=np.float32)
    for new_i, original_i in enumerate(used_indices):
        reduced[original_i] = reduced_partial[new_i]

    pair_features = np.array([
        np.concatenate((reduced[i], reduced[j]))
        for i, j in tqdm(combinations, desc="Combining pairs")
    ])

    os.makedirs('AE_CNN_Att_MLP_Hybrid_results_1000_sequential/Hybrid_Attention', exist_ok=True)
    out_dir = 'AE_CNN_Att_MLP_Hybrid_results_1000_sequential/Hybrid_Attention'

    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    weight_dict = dict(zip(np.unique(labels), weights))

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    all_metrics = {k: [] for k in ["Micro", "Macro", "Weighted"]}

    for fold, (train_idx, test_idx) in enumerate(kf.split(pair_features, labels), 1):
        print(f"\n--- Fold {fold} ---")
        X_train, y_train = pair_features[train_idx], labels[train_idx]
        X_test, y_test = pair_features[test_idx], labels[test_idx]

        model = build_hybrid_model((encoding_dim * 2,), num_classes)
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs_mlp,
            class_weight=weight_dict,
            verbose=0,
            callbacks=[
                TqdmCallback(verbose=1),
                TerminateOnNaN(),
                EarlyStopping(patience=patience, restore_best_weights=True)
            ]
        )

        y_probs = model.predict(X_test)
        y_pred = np.argmax(y_probs, axis=1)
        metrics = compute_metrics(y_test, y_pred, y_probs, num_classes)

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        fold_dir = os.path.join(out_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        for key in metrics:
            with open(os.path.join(fold_dir, f'metrics_{key}.txt'), 'w') as f:
                for m, val in metrics[key].items():
                    f.write(f"{m}: {val:.4f}\n")
        model.save(os.path.join(fold_dir, 'model.h5'))
        K.clear_session()

    with open(os.path.join(out_dir, 'average_metrics.txt'), 'w') as f:
        for key in all_metrics:
            for metric in all_metrics[key][0]:
                avg = np.mean([m[metric] for m in all_metrics[key]])
                std = np.std([m[metric] for m in all_metrics[key]])
                f.write(f"{key} {metric}: {avg:.4f} ± {std:.4f}\n")

if __name__ == "__main__":
    main()