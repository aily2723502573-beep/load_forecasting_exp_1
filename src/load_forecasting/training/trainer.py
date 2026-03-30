import copy
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..models.factory import FLATTEN_INPUT_MODELS, TREE_MODELS, build_model


class ModelTrainer:
    def __init__(
        self,
        device: torch.device,
        patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        verbose: bool = False,
        log_every: int = 1,
        model_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        self.log_every = log_every
        self.config = config or {}
        self.model_config = model_config or self.config.get("models", {})
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_model_weights: Optional[dict] = None

    def _build_model(
        self,
        model_type: str,
        input_dim: int,
        output_dim: int,
        window_size: int,
        horizon_steps: int,
    ):
        cfg = dict(self.model_config.get(model_type, {}))
        if model_type in TREE_MODELS:
            seed = self.config.get("seed", 42)
            cfg.setdefault("random_seed", seed)
            cfg.setdefault("random_state", seed)
        net = build_model(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            window_size=window_size,
            horizon_steps=horizon_steps,
            model_config=cfg,
        )
        #lr = self.config.get("training", {}).get("learning_rate", 1e-3)
        lr = float(self.config.get("training", {}).get("learning_rate", 1e-3))
        
        return net, lr

    def _train_tree_model(
        self,
        dataloader_train,
        building_name: str,
        model_type: str,
        input_dim: int,
        output_dim: int,
        window_size: int,
        horizon_steps: int,
    ):
        """Train CatBoost or XGBoost with fit()."""
        import joblib

        start_time = time.time()
        model, _ = self._build_model(model_type, input_dim, output_dim, window_size, horizon_steps)

        def to_numpy(dl):
            X_list, y_list = [], []
            for dp, lb in dl:
                dp_flat = dp.reshape(dp.size(0), -1).detach().cpu().numpy()
                X_list.append(dp_flat)
                y_list.append(lb.detach().cpu().numpy())
            return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

        X_train, y_train = to_numpy(dataloader_train)

        model.fit(X_train, y_train)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"{building_name}_{model_type}_len{horizon_steps}.pkl")
        joblib.dump(model, path)
        return model, time.time() - start_time

    def train(
        self,
        dataloader_train,
        dataloader_val,
        building_name: str,
        epochs: int,
        model_type: str,
        input_dim: int,
        output_dim: int,
        window_size: int,
        horizon_steps: int,
    ):
        start_time = time.time()
        # Reset early-stopping state at the start of each training run
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None

        if model_type in TREE_MODELS:
            return self._train_tree_model(
                dataloader_train, building_name, model_type,
                input_dim, output_dim, window_size, horizon_steps,
            )

        net, lr = self._build_model(model_type, input_dim, output_dim, window_size, horizon_steps)
        net = net.to(self.device)

        criterion = nn.MSELoss()
        #optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5)

        for t in range(1, epochs + 1):
            if self.early_stop:
                break

            net.train()
            train_loss = 0.0
            train_samples = 0
            for _, (datapoints, labels) in enumerate(dataloader_train):
                datapoints, labels = datapoints.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                if model_type in FLATTEN_INPUT_MODELS:
                    preds = net(datapoints.reshape(datapoints.shape[0], -1))
                else:
                    preds = net(datapoints)
                loss = criterion(preds, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size
                train_samples += batch_size
                del datapoints, labels, preds, loss

            net.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for _, (datapoints, labels) in enumerate(dataloader_val):
                    datapoints, labels = datapoints.to(self.device), labels.to(self.device)
                    if model_type in FLATTEN_INPUT_MODELS:
                        preds = net(datapoints.reshape(datapoints.shape[0], -1))
                    else:
                        preds = net(datapoints)
                    batch_size = labels.size(0)
                    val_loss += criterion(preds, labels).item() * batch_size
                    val_samples += batch_size
                    del datapoints, labels, preds

            avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
            avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
            scheduler.step(avg_val_loss)

            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_model_weights = copy.deepcopy(net.state_dict())
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

            if self.verbose and (t % self.log_every == 0 or t == 1 or self.early_stop):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {t}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | Best Val Loss: {self.best_loss:.6f}"
                )
                if self.early_stop:
                    print(f"Early stopping triggered at epoch {t}")

        if self.best_model_weights is not None:
            net.load_state_dict(self.best_model_weights)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path_by_len = os.path.join(self.checkpoint_dir, f"{building_name}_{model_type}_len{horizon_steps}.pt")
        torch.save(net.state_dict(), path_by_len)

        training_time = time.time() - start_time
        return net, training_time
