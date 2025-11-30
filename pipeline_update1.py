import argparse
import importlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Optional


DEFAULT_INPUT_DIR = Path("/System/Volumes/Data/Users/yuqitao/Downloads/PBM_65536")
TF_NAMES = [
    "CUX1",
    "CUX2",
    "MAX",
    "DLX3",
    "NKX2.5",
    "NFATC2",
    "POU5F1",
    "LHX9",
]
BASE_ORDER = ["A", "C", "G", "T"]
NT_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class TrainingResult:
    best_model_path: Path
    history: Dict[str, List[float]]
    metrics: Dict[str, float]
    scatter_path: Optional[Path] = None


@dataclass
class UnifiedDatasetSplits:
    train: "NTDataset"
    val: "NTDataset"
    test: "NTDataset"
    tokenizer_name: str


# Data Loading

def read_zscore_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["kmer", "z"], engine="python")
    return df


def load_tf_frames(tf_name: str, input_dir: Path) -> pd.DataFrame:
    cytosine_path = next(input_dir.glob(f"*{tf_name}*Cytosine.txt"))
    methyl_path = next(input_dir.glob(f"*{tf_name}*5mCG.txt"))
    cyt = read_zscore_file(cytosine_path)
    methyl = read_zscore_file(methyl_path)
    merged = (
        cyt.rename(columns={"z": "z_cytosine"})
        .merge(methyl.rename(columns={"z": "z_5mcg"}), on="kmer", how="inner")
        .dropna()
    )
    merged["delta_score"] = merged["z_5mcg"] - merged["z_cytosine"]
    return merged


def load_all_tf_frames(tf_names: List[str], input_dir: Path) -> pd.DataFrame:
    frames = []
    for tf_name in tf_names:
        tf_frame = load_tf_frames(tf_name, input_dir)
        tf_frame["tf_name"] = tf_name
        frames.append(tf_frame)
    return pd.concat(frames, ignore_index=True)


# Encoding

def encode_kmer(kmer: str) -> np.ndarray:
    array = np.zeros((len(kmer), len(BASE_ORDER)), dtype=np.float32)
    base_index = {base: idx for idx, base in enumerate(BASE_ORDER)}
    for pos, base in enumerate(kmer.upper()):
        idx = base_index.get(base)
        if idx is not None:
            array[pos, idx] = 1.0
    return array.reshape(-1)


def build_encoded_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack([encode_kmer(kmer) for kmer in df["kmer"]])
    y = df["delta_score"].to_numpy(dtype=np.float32)
    return X, y


def compute_cpg_mask(kmer: str, max_len: int) -> np.ndarray:
    mask = np.zeros(max_len, dtype=np.int64)
    for idx, base in enumerate(kmer.upper()):
        if idx + 1 < len(kmer) and base == "C" and kmer[idx + 1].upper() == "G":
            mask[idx] = 1
    return mask


# Splitting

def split_dataset(X: np.ndarray, y: np.ndarray, seed: int = 42) -> DatasetSplits:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    n_total = len(X)
    train_end = math.floor(0.7 * n_total)
    val_end = train_end + math.floor(0.15 * n_total)

    X_train, y_train = X_shuffled[:train_end], y_shuffled[:train_end]
    X_val, y_val = X_shuffled[train_end:val_end], y_shuffled[train_end:val_end]
    X_test, y_test = X_shuffled[val_end:], y_shuffled[val_end:]
    return DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test)


# Model

def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
    )


# Training

def create_dataloaders(splits: DatasetSplits, batch_size: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(splits.X_train), torch.from_numpy(splits.y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(splits.X_val), torch.from_numpy(splits.y_val).unsqueeze(1))
    test_ds = TensorDataset(torch.from_numpy(splits.X_test), torch.from_numpy(splits.y_test).unsqueeze(1))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def train_model(
    model: nn.Module,
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 5,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    train_loader, val_loader, _ = loaders
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    epochs_no_improve = 0
    history = {"train": [], "val": []}

    model.to(device)
    best_state = model.state_dict()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        history["train"].append(avg_train)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_X)
                val_losses.append(criterion(preds, batch_y).item())
        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        history["val"].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def train_nt_model(
    model: nn.Module,
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 20,
    patience: int = 3,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    train_loader, val_loader, _ = loaders
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    epochs_no_improve = 0
    history = {"train": [], "val": []}

    model.to(device)
    best_state = model.state_dict()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            methyl_mask = batch["methyl_mask"].to(device)
            labels = batch["label"].to(device)

            preds = model(input_ids=input_ids, attention_mask=attention_mask, methyl_mask=methyl_mask)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        history["train"].append(avg_train)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                methyl_mask = batch["methyl_mask"].to(device)
                labels = batch["label"].to(device)
                preds = model(input_ids=input_ids, attention_mask=attention_mask, methyl_mask=methyl_mask)
                val_losses.append(criterion(preds, labels).item())

        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        history["val"].append(avg_val)

        logging.info("Epoch %d | train=%.4f | val=%.4f", epoch, avg_train, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


# Evaluation 

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).cpu().numpy().flatten()
            preds_list.append(outputs)
            targets_list.append(batch_y.numpy().flatten())
    return np.concatenate(preds_list), np.concatenate(targets_list)


def evaluate_nt(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    tf_list: List[str] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            methyl_mask = batch["methyl_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, methyl_mask=methyl_mask)
            preds_list.append(outputs.cpu().numpy().flatten())
            targets_list.append(batch["label"].numpy().flatten())
            tf_list.extend(batch["tf"])
    return np.concatenate(preds_list), np.concatenate(targets_list), tf_list


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    pearson_r = float(pearsonr(preds, targets).statistic) if len(preds) > 1 else float("nan")
    spearman_rho = float(spearmanr(preds, targets).statistic) if len(preds) > 1 else float("nan")
    r2 = float(r2_score(targets, preds)) if len(preds) > 1 else float("nan")
    return {"pearson_r": pearson_r, "spearman_rho": spearman_rho, "r2": r2}


def compute_per_tf_metrics(preds: np.ndarray, targets: np.ndarray, tf_names: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    unique_tfs = sorted(set(tf_names))
    for tf_name in unique_tfs:
        mask = np.array([name == tf_name for name in tf_names])
        tf_preds = preds[mask]
        tf_targets = targets[mask]
        metrics[tf_name] = compute_metrics(tf_preds, tf_targets)
    return metrics


def plot_scatter(preds: np.ndarray, targets: np.ndarray, path: Path, sample_size: int = 200) -> None:
    if len(preds) == 0:
        return
    rng = np.random.default_rng(0)
    if len(preds) > sample_size:
        indices = rng.choice(len(preds), size=sample_size, replace=False)
        preds = preds[indices]
        targets = targets[indices]

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.6, edgecolor="k", linewidths=0.3)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="y=x")
    plt.xlabel("True Δscore")
    plt.ylabel("Predicted Δscore")
    plt.title("Predicted vs True")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# Nucleotide Transformer helpers


class NTDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        methyl_masks: torch.Tensor,
        labels: torch.Tensor,
        tf_names: List[str],
        tokenizer_name: str = "",
    ) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.methyl_masks = methyl_masks
        self.labels = labels
        self.tf_names = tf_names
        self.tokenizer_name = tokenizer_name

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "methyl_mask": self.methyl_masks[idx],
            "label": self.labels[idx],
            "tf": self.tf_names[idx],
        }

    def subset(self, indices: np.ndarray) -> "NTDataset":
        return NTDataset(
            self.input_ids[indices],
            self.attention_masks[indices],
            self.methyl_masks[indices],
            self.labels[indices],
            [self.tf_names[i] for i in indices],
            tokenizer_name=self.tokenizer_name,
        )


def load_nt_components(model_name: str):
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    backbone = transformers.AutoModel.from_pretrained(model_name)
    return tokenizer, backbone


def build_nt_dataset(df: pd.DataFrame, tokenizer) -> NTDataset:
    sequences = df["kmer"].tolist()
    tf_names = df["tf_name"].tolist()
    labels = torch.tensor(df["delta_score"].to_numpy(dtype=np.float32)).unsqueeze(1)

    encoded = tokenizer(
        sequences,
        add_special_tokens=False,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    max_len = encoded["input_ids"].shape[1]
    methyl_masks = np.stack([compute_cpg_mask(seq, max_len) for seq in sequences])

    return NTDataset(
        input_ids=encoded["input_ids"],
        attention_masks=encoded["attention_mask"],
        methyl_masks=torch.tensor(methyl_masks, dtype=torch.long),
        labels=labels,
        tf_names=tf_names,
        tokenizer_name=getattr(tokenizer, "name_or_path", ""),
    )


def split_unified_dataset(dataset: NTDataset, seed: int = 42) -> UnifiedDatasetSplits:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    n_total = len(indices)
    train_end = math.floor(0.7 * n_total)
    val_end = train_end + math.floor(0.15 * n_total)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return UnifiedDatasetSplits(
        train=dataset.subset(train_idx),
        val=dataset.subset(val_idx),
        test=dataset.subset(test_idx),
        tokenizer_name=dataset.tokenizer_name,
    )


def create_nt_dataloaders(splits: UnifiedDatasetSplits, batch_size: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(splits.train, batch_size=batch_size, shuffle=True),
        DataLoader(splits.val, batch_size=batch_size, shuffle=False),
        DataLoader(splits.test, batch_size=batch_size, shuffle=False),
    )


class NucleotideTransformerRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.methyl_embed = nn.Embedding(2, hidden_size)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, methyl_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        methyl_emb = self.methyl_embed(methyl_mask)
        hidden = hidden + methyl_emb
        masked = hidden * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return self.regressor(pooled)


# Pipeline 

def ensure_dirs(output_dir: Path, tf_name: str) -> Dict[str, Path]:
    tf_root = output_dir / tf_name
    paths = {
        "tf_root": tf_root,
        "cache": tf_root / "cache",
        "models": tf_root / "models",
        "plots": tf_root / "plots",
        "metrics": tf_root / "metrics",
        "splits": tf_root / "splits",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_nt_dirs(output_dir: Path) -> Dict[str, Path]:
    nt_root = output_dir / "shared_nt"
    paths = {
        "tf_root": nt_root,
        "cache": nt_root / "cache",
        "models": nt_root / "models",
        "plots": nt_root / "plots",
        "metrics": nt_root / "metrics",
        "splits": nt_root / "splits",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def cache_arrays(paths: Dict[str, Path], tf_name: str, X: np.ndarray, y: np.ndarray) -> Path:
    cache_path = paths["cache"] / f"{tf_name}_arrays.npz"
    np.savez_compressed(cache_path, X=X, y=y)
    return cache_path


def save_splits(paths: Dict[str, Path], tf_name: str, splits: DatasetSplits) -> None:
    np.savez_compressed(
        paths["splits"] / f"{tf_name}_splits.npz",
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
    )


def save_history(paths: Dict[str, Path], tf_name: str, history: Dict[str, List[float]]) -> Path:
    history_path = paths["metrics"] / f"{tf_name}_history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    return history_path


def save_metrics(paths: Dict[str, Path], tf_name: str, metrics: Dict[str, float]) -> Path:
    metrics_path = paths["metrics"] / f"{tf_name}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path


def save_model(paths: Dict[str, Path], tf_name: str, model: nn.Module) -> Path:
    model_path = paths["models"] / f"{tf_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    return model_path


def process_tf(
    tf_name: str,
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
    plot: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 5,
) -> TrainingResult:
    logging.info("Processing TF %s", tf_name)
    paths = ensure_dirs(output_dir, tf_name)
    df = load_tf_frames(tf_name, input_dir)
    X, y = build_encoded_arrays(df)
    cache_arrays(paths, tf_name, X, y)

    splits = split_dataset(X, y, seed)
    save_splits(paths, tf_name, splits)

    model = build_model()
    loaders = create_dataloaders(splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, history = train_model(
        model,
        loaders,
        device,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
    )
    save_history(paths, tf_name, history)

    _, _, test_loader = loaders
    preds, targets = evaluate(model, test_loader, device)
    metrics = compute_metrics(preds, targets)
    metrics_path = save_metrics(paths, tf_name, metrics)

    model_path = save_model(paths, tf_name, model)

    scatter_path = None
    if plot:
        scatter_path = paths["plots"] / f"{tf_name}_scatter.png"
        plot_scatter(preds, targets, scatter_path)

    logging.info(
        "Finished TF %s | metrics saved to %s | model saved to %s",
        tf_name,
        metrics_path,
        model_path,
    )

    return TrainingResult(best_model_path=model_path, history=history, metrics=metrics, scatter_path=scatter_path)


def process_nt_model(
    tf_names: List[str],
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
    model_name: str = NT_MODEL_NAME,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 20,
    patience: int = 3,
    plot: bool = True,
) -> Dict[str, object]:
    logging.info("Loading unified dataset for TFs: %s", ", ".join(tf_names))
    paths = ensure_nt_dirs(output_dir)
    df = load_all_tf_frames(tf_names, input_dir)
    df.to_csv(paths["cache"] / "unified_dataset.tsv", sep="\t", index=False)

    tokenizer, backbone = load_nt_components(model_name)
    dataset = build_nt_dataset(df, tokenizer)
    splits = split_unified_dataset(dataset, seed)
    loaders = create_nt_dataloaders(splits, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = getattr(backbone.config, "hidden_size", 256)
    model = NucleotideTransformerRegressor(backbone, hidden_size=hidden_size, dropout=0.1)

    logging.info("Starting fine-tuning of Nucleotide Transformer (%s) on %d samples", model_name, len(dataset))
    model, history = train_nt_model(
        model,
        loaders,
        device,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
    )
    history_path = save_history(paths, "shared", history)

    _, _, test_loader = loaders
    preds, targets, tf_list = evaluate_nt(model, test_loader, device)
    overall_metrics = compute_metrics(preds, targets)
    per_tf_metrics = compute_per_tf_metrics(preds, targets, tf_list)

    metrics = {"overall": overall_metrics, "per_tf": per_tf_metrics}
    metrics_path = paths["metrics"] / "shared_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    model_path = paths["models"] / "shared_nt_model.pt"
    torch.save({"state_dict": model.state_dict(), "model_name": model_name}, model_path)

    scatter_paths: Dict[str, Path] = {}
    if plot:
        for tf_name in tf_names:
            tf_mask = np.array([name == tf_name for name in tf_list])
            tf_preds = preds[tf_mask]
            tf_targets = targets[tf_mask]
            scatter_path = paths["plots"] / f"{tf_name}_scatter.png"
            plot_scatter(tf_preds, tf_targets, scatter_path)
            scatter_paths[tf_name] = scatter_path

    logging.info("Finished unified model | metrics saved to %s | model saved to %s", metrics_path, model_path)

    return {
        "model_path": model_path,
        "history_path": history_path,
        "metrics_path": metrics_path,
        "scatter_paths": scatter_paths,
        "overall_metrics": overall_metrics,
        "per_tf_metrics": per_tf_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train methylation delta predictors for TFs.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing *_Zscore_Cytosine.txt and *_Zscore_5mCG.txt files.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory to store cached arrays, splits, metrics, and models.")
    parser.add_argument("--tfs", nargs="*", default=TF_NAMES, help="TF names to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs across TFs. Use -1 for all cores.")
    parser.add_argument("--no_plot", action="store_true", help="Disable scatter plot generation.")
    parser.add_argument("--model_type", choices=["mlp", "nt"], default="nt", help="Choose between per-TF MLPs or a shared Nucleotide Transformer model.")
    parser.add_argument("--nt_model_name", type=str, default=NT_MODEL_NAME, help="Hugging Face model name for the Nucleotide Transformer backbone.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot = not args.no_plot

    if args.model_type == "nt":
        result = process_nt_model(
            args.tfs,
            args.input_dir,
            args.output_dir,
            seed=args.seed,
            model_name=args.nt_model_name,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            plot=plot,
        )
        summary_path = args.output_dir / "metrics_summary.json"
        with summary_path.open("w") as f:
            json.dump({"overall": result["overall_metrics"], "per_tf": result["per_tf_metrics"]}, f, indent=2)
        logging.info("Wrote unified model metrics to %s", summary_path)
    else:
        if args.n_jobs and args.n_jobs != 1:
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_tf)(
                    tf,
                    args.input_dir,
                    args.output_dir,
                    args.seed,
                    plot,
                    args.lr,
                    args.weight_decay,
                    args.max_epochs,
                    args.patience,
                )
                for tf in args.tfs
            )
        else:
            results = [
                process_tf(
                    tf,
                    args.input_dir,
                    args.output_dir,
                    args.seed,
                    plot,
                    args.lr,
                    args.weight_decay,
                    args.max_epochs,
                    args.patience,
                )
                for tf in args.tfs
            ]

        summary = {tf: result.metrics for tf, result in zip(args.tfs, results)}
        summary_path = args.output_dir / "metrics_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        logging.info("Wrote per-TF MLP summary metrics to %s", summary_path)


if __name__ == "__main__":
    main()