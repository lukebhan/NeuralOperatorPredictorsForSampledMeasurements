"""Training utilities for the multistep predictor model (Case 2)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import copy
import time
import matplotlib.pyplot as plt

def build_multistep_fno_dataset(dataset):
    """Tile the state across the delay grid and concatenate with u_hist to form FNO inputs.

    Inputs:  dataset dict with "state" (N, state_dim), "u_hist" (N, delay_steps, nv),
             "predictor_traj" (N, horizon, state_dim)
    Returns: X (N, delay_steps, state_dim+nv), Y (N, horizon, state_dim)
    """
    state = dataset["state"]                   # (N, state_dim)
    u_hist = dataset["u_hist"]                 # (N, delay_steps, control_dim)
    target = dataset["predictor_traj"]         # (N, horizon, state_dim)

    N, delay_steps, _ = u_hist.shape

    state_rep = np.repeat(state[:, None, :], delay_steps, axis=1)
    X = np.concatenate([state_rep, u_hist], axis=2)

    return X, target

def fit_multistep_normalizers(X_train, Y_train, eps=1e-8):
    """Compute per-channel mean and std for normalizing multistep inputs and trajectories.

    Inputs:  X_train (N, delay_steps, channels), Y_train (N, horizon, state_dim), eps float
    Returns: stats dict with "x_mean", "x_std", "y_mean", "y_std"
    """
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True) + eps

    y_mean = Y_train.mean(axis=(0, 1), keepdims=True)
    y_std = Y_train.std(axis=(0, 1), keepdims=True) + eps

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    return stats

def normalize_multistep_dataset(X, Y, stats):
    """Normalize X and Y using the provided stats dict.

    Inputs:  X, Y arrays; stats dict from fit_multistep_normalizers
    Returns: (Xn, Yn) normalized arrays
    """
    Xn = (X - stats["x_mean"]) / stats["x_std"]
    Yn = (Y - stats["y_mean"]) / stats["y_std"]

    return Xn, Yn

def denormalize_multistep_predictions(Yn, stats):
    """Invert output normalization to recover trajectory predictions in original units."""
    return Yn * stats["y_std"] + stats["y_mean"]

def make_multistep_dataloaders(
    X,
    Y,
    batch_size=64,
    val_fraction=0.2,
    seed=0,
):
    """Split data, fit normalizers, and return train/val DataLoaders.

    Inputs:  X, Y arrays; batch_size, val_fraction, seed
    Returns: (train_loader, val_loader, stats)
    """
    X_train, X_val, Y_train, Y_val = train_test_split(
        X,
        Y,
        test_size=val_fraction,
        random_state=seed,
    )

    stats = fit_multistep_normalizers(X_train, Y_train)

    X_train, Y_train = normalize_multistep_dataset(X_train, Y_train, stats)
    X_val, Y_val = normalize_multistep_dataset(X_val, Y_val, stats)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, stats

def train_multistep_one_epoch(model, loader, optimizer, device):
    """Run one training epoch and return the mean MSE loss."""
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)                     # (B, horizon, state_dim)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = xb.shape[0]
        # Accumulate sum-of-losses so we get a sample-weighted average.
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def evaluate_multistep_model(model, loader, device):
    """Evaluate the model on a DataLoader and return the mean MSE loss."""
    model.eval()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)                     # (B, horizon, state_dim)
        loss = loss_fn(pred, yb)

        batch_size = xb.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def train_multistep_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-6,
    save_path="multistep_predictor_fno.pt",
):
    """Train with Adam + ReduceLROnPlateau, saving the best checkpoint.

    Inputs:  model, train_loader, val_loader, device, epochs, lr, weight_decay, save_path
    Returns: (model with best weights, history dict with "train_loss" / "val_loss")
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_multistep_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )
        val_loss = evaluate_multistep_model(
            model,
            val_loader,
            device,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_time = time.perf_counter() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "epoch": epoch,
                },
                save_path,
            )

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:4d} | "
            f"train {train_loss:.6e} | "
            f"val {val_loss:.6e} | "
            f"lr {lr_now:.2e} | "
            f"time {epoch_time:.2f}s"
        )

    # Strip internal _metadata key before loading (see case1_trainer for rationale).
    clean_state = {k: v for k, v in best_state.items()}
    clean_state.pop("_metadata", None)

    model.load_state_dict(clean_state)
    return model, history


def load_trained_multistep_model(model, checkpoint_path, device):
    """Load a checkpoint into model and set it to eval mode.

    Inputs:  model, checkpoint_path string, device
    Returns: (model in eval mode, full checkpoint dict)
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]
    state_dict.pop("_metadata", None)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, checkpoint


def plot_multistep_training_history(history):
    """Plot training and validation MSE loss curves on a log scale."""
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Multistep predictor training history")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()