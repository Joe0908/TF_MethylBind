import argparse
import json
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
from torch.utils.data import DataLoader, TensorDataset


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
    scatter_path: Path | None


# ------------------------- Data Loading ------------------------- #

def read_zscore_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["kmer", "z"], engine="python")
    return df


def load_tf_frames(tf_name: str, input_dir: Path) -> pd.DataFrame:
    cytosine_path = input_dir / f"{tf_name}_Zscore_Cytosine.txt"
    methyl_path = input_dir / f"{tf_name}_Zscore_5mCG.txt"
    cyt = read_zscore_file(cytosine_path)
    methyl = read_zscore_file(methyl_path)
    merged = (
        cyt.rename(columns={"z": "z_cytosine"})
        .merge(methyl.rename(columns={"z": "z_5mcg"}), on="kmer", how="inner")
        .dropna()
    )
    merged["delta_score"] = merged["z_5mcg"] - merged["z_cytosine"]
    return merged


# ------------------------- Encoding ------------------------- #

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


# ------------------------- Splitting ------------------------- #

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


# ------------------------- Model ------------------------- #

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


# ------------------------- Training ------------------------- #

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


# ------------------------- Evaluation ------------------------- #

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


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    pearson_r = float(pearsonr(preds, targets).statistic) if len(preds) > 1 else float("nan")
    spearman_rho = float(spearmanr(preds, targets).statistic) if len(preds) > 1 else float("nan")
    r2 = float(r2_score(targets, preds)) if len(preds) > 1 else float("nan")
    return {"pearson_r": pearson_r, "spearman_rho": spearman_rho, "r2": r2}


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


# ------------------------- Pipeline ------------------------- #

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


def process_tf(tf_name: str, input_dir: Path, output_dir: Path, seed: int = 42, plot: bool = True) -> TrainingResult:
    paths = ensure_dirs(output_dir, tf_name)
    df = load_tf_frames(tf_name, input_dir)
    X, y = build_encoded_arrays(df)
    cache_arrays(paths, tf_name, X, y)

    splits = split_dataset(X, y, seed)
    save_splits(paths, tf_name, splits)

    model = build_model()
    loaders = create_dataloaders(splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, history = train_model(model, loaders, device)
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

    return TrainingResult(best_model_path=model_path, history=history, metrics=metrics, scatter_path=scatter_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train methylation delta predictors for TFs.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing *_Zscore_Cytosine.txt and *_Zscore_5mCG.txt files.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory to store cached arrays, splits, metrics, and models.")
    parser.add_argument("--tfs", nargs="*", default=TF_NAMES, help="TF names to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs across TFs. Use -1 for all cores.")
    parser.add_argument("--no_plot", action="store_true", help="Disable scatter plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot = not args.no_plot

    if args.n_jobs and args.n_jobs != 1:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_tf)(tf, args.input_dir, args.output_dir, args.seed, plot) for tf in args.tfs
        )
    else:
        results = [process_tf(tf, args.input_dir, args.output_dir, args.seed, plot) for tf in args.tfs]

    summary = {tf: result.metrics for tf, result in zip(args.tfs, results)}
    summary_path = args.output_dir / "metrics_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
