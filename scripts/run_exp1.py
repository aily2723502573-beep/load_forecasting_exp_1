
import argparse
import os
import time  # NEW

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

from load_forecasting.data.windowing import create_windowed_loader
from load_forecasting.evaluation import (
    ModelTester,
    PERSISTENCE_MODEL,
    compute_metrics,
    compute_weighted_mape,
)
from load_forecasting.training.trainer import ModelTrainer
from load_forecasting.utils.reproducibility import set_seed


# NEW: simple logger
def log(msg: str):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="EXP 1: Univariate train/eval (Power only) - Multi-horizon Building Load Forecasting"
    )
    parser.add_argument("--config", default="configs/exp1.yaml")
    parser.add_argument("--building", default="office")
    parser.add_argument("--data_dir", default=None, help="Data directory (defaults to config.data.raw_data_dir)")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=range(1,30),
        help="Forecast horizons in days (converted to hourly steps)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch training logs")
    parser.add_argument("--log_every", type=int, default=1, help="Log every N epochs when --verbose is set")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use. Use 'cpu' if GPU (e.g. RTX 50 series) is incompatible with current PyTorch.",
    )
    args = parser.parse_args()

    # ---------------- device + seed (keep only one copy) ----------------
    set_seed(args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        log(f"Using GPU: {torch.cuda.get_device_name(0)} | CUDA={torch.version.cuda} | n_gpus={torch.cuda.device_count()}")
    else:
        log("Using CPU")
    log(f"Device = {device}")

    # ---------------- load config ----------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, args.config)
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed
    exp_name = os.path.splitext(os.path.basename(args.config))[0]

    log(f"Config path: {config_path}")
    log(f"Experiment name: {exp_name}")

    # ---------------- load data ----------------
    data_dir = args.data_dir or os.path.join(base_dir, config["data"]["raw_data_dir"])
    building = args.building
    csv_path = os.path.join(data_dir, f"{building}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Please place {building}.csv in {data_dir} with columns: Time, Power (kW)"
        )

    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

    log(f"Loaded data: {csv_path}")
    log(f"Total rows: {len(df)} | Time range: {df['Time'].min()} ~ {df['Time'].max()}")

    input_features = ["Power (kW)"]
    output_features = ["Power (kW)"]

    # ---------------- split ----------------
    n = len(df)
    train_end = int(0.7 * n)
    val_end = train_end + int(0.1 * n)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    log(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ---------------- scale ----------------
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    train_x = scaler_x.fit_transform(np.array(train_df[input_features]))
    val_x = scaler_x.transform(np.array(val_df[input_features]))
    test_x = scaler_x.transform(np.array(test_df[input_features]))

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y = scaler_y.fit_transform(np.array(train_df[output_features]))
    val_y = scaler_y.transform(np.array(val_df[output_features]))
    test_y = scaler_y.transform(np.array(test_df[output_features]))

    # ---------------- training settings ----------------
    window_size = config["training"]["window_size"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience", 10)

    input_dim = 1
    output_dim = 1

    model_types = config["models"]["model_types"]

    checkpoint_dir = os.path.join(base_dir, args.checkpoint_dir, exp_name)
    out_dir = os.path.join(base_dir, args.out_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    log(f"window_size={window_size} | batch_size={batch_size} | epochs={epochs} | patience={patience}")
    log(f"horizons(days)={args.horizons}")
    log(f"models={model_types}")
    log(f"checkpoint_dir={checkpoint_dir}")
    log(f"out_dir={out_dir}")

    # ---------------- run ----------------
    all_rows = []
    for hi, days in enumerate(args.horizons, start=1):
        horizon_steps = int(days) * 24
        log(f"==================== Horizon {hi}/{len(args.horizons)}: {days} days ({horizon_steps} steps) ====================")

        # create windowed data
        dl_train, _, _ = create_windowed_loader(window_size, horizon_steps, batch_size, train_x, train_y)
        dl_val, X_val, y_val = create_windowed_loader(window_size, horizon_steps, batch_size, val_x, val_y)
        _, X_test, y_test = create_windowed_loader(window_size, horizon_steps, batch_size, test_x, test_y)

        log(f"Prepared windows: X_val={X_val.shape}, y_val={y_val.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

        for mi, model_type in enumerate(model_types, start=1):
            log(f"---- Model {mi}/{len(model_types)}: {model_type} | horizon={days}d ----")

            train_time_sec = 0.0

            # train
            if model_type != PERSISTENCE_MODEL:
                log(f"Training start: {model_type}")
                t0 = time.time()

                trainer = ModelTrainer(
                    device=device,
                    patience=patience,
                    checkpoint_dir=checkpoint_dir,
                    verbose=args.verbose,
                    log_every=args.log_every,
                    config=config,
                )
                _, train_time_sec = trainer.train(
                    dl_train,
                    dl_val,
                    building,
                    epochs,
                    model_type,
                    input_dim,
                    output_dim,
                    window_size,
                    horizon_steps,
                )

                log(f"Training done: {model_type} | train_time_sec={train_time_sec:.2f}s | wall={time.time()-t0:.2f}s")
            else:
                log("Skip training (Persistence baseline).")

            # eval
            log(f"Eval start: {model_type}")

            tester = ModelTester(
                input_dim,
                window_size,
                output_dim,
                building,
                device=device,
                checkpoint_dir=checkpoint_dir,
                model_config=config.get("models", {}),
            )

            y_val_pred, y_val_true = tester.test(X_val, y_val, horizon_steps, model_type, scaler_y)
            y_test_pred, y_test_true = tester.test(X_test, y_test, horizon_steps, model_type, scaler_y)

            val_mae, val_rmse, val_r2, val_mape = compute_metrics(y_val_true.flatten(), y_val_pred.flatten())
            test_mae, test_rmse, test_r2, test_mape = compute_metrics(y_test_true.flatten(), y_test_pred.flatten())

            val_wmape = compute_weighted_mape(y_val_true, y_val_pred, horizon_steps)
            test_wmape = compute_weighted_mape(y_test_true, y_test_pred, horizon_steps)

            # always print a one-line summary (even if not --verbose)
            log(
                f"Scores: {model_type} | {days}d | "
                f"VAL(MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R2={val_r2:.4f}, wMAPE={val_wmape:.4f}) | "
                f"TEST(MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}, wMAPE={test_wmape:.4f})"
            )

            all_rows.append(
                {
                    "seed": args.seed,
                    "building": building,
                    "model": model_type,
                    "window_size": window_size,
                    "horizon_days": days,
                    "horizon_steps": horizon_steps,
                    "x_scaler": "MinMaxScaler",
                    "y_scaler": "MinMaxScaler",
                    "scaler_range": "(0, 1)",
                    "train_time_sec": train_time_sec,
                    "split": "val",
                    "MAE": val_mae,
                    "RMSE": val_rmse,
                    "R2": val_r2,
                    "MAPE": val_mape,
                    "wMAPE": val_wmape,
                }
            )
            all_rows.append(
                {
                    "seed": args.seed,
                    "building": building,
                    "model": model_type,
                    "window_size": window_size,
                    "horizon_days": days,
                    "horizon_steps": horizon_steps,
                    "x_scaler": "MinMaxScaler",
                    "y_scaler": "MinMaxScaler",
                    "scaler_range": "(0, 1)",
                    "train_time_sec": train_time_sec,
                    "split": "test",
                    "MAE": test_mae,
                    "RMSE": test_rmse,
                    "R2": test_r2,
                    "MAPE": test_mape,
                    "wMAPE": test_wmape,
                }
            )

        log(f"==================== Horizon {days} days finished ====================")

    # ---------------- save ----------------
    df_out = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, f"{building}_scores.csv")
    df_out.to_csv(out_path, index=False)
    log(f"Saved: {out_path}")

    # OPTIONAL: print best model per horizon by TEST MAE
    try:
        df_test = df_out[df_out["split"] == "test"].copy()
        best = df_test.loc[df_test.groupby("horizon_days")["MAE"].idxmin()][
            ["horizon_days", "model", "MAE", "RMSE", "R2", "MAPE", "wMAPE", "train_time_sec"]
        ].sort_values("horizon_days")
        log("Best model per horizon (by TEST MAE):")
        print(best.to_string(index=False))
    except Exception as e:
        log(f"[warn] failed to compute best summary: {e}")


if __name__ == "__main__":
    main()


