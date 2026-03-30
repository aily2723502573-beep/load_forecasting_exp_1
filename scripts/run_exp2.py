import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

from load_forecasting.data.windowing import create_windowed_loader
from load_forecasting.evaluation import ModelTester, PERSISTENCE_MODEL, compute_metrics, compute_weighted_mape
from load_forecasting.training.trainer import ModelTrainer
from load_forecasting.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="EXP 2: Multivariate train/eval (multi-source features) - Multi-horizon Building Load Forecasting"
    )
    parser.add_argument("--config", default="configs/exp2.yaml")
    parser.add_argument("--building", default="office")
    parser.add_argument("--data_dir", default=None, help="Data directory (defaults to config.data.raw_data_dir)")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=range(1,31),
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

    set_seed(args.seed)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed

    exp_name = os.path.splitext(os.path.basename(args.config))[0]

    data_dir = args.data_dir or os.path.join(base_dir, config["data"]["raw_data_dir"])
    building = args.building
    csv_path = os.path.join(data_dir, f"{building}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Please place {building}.csv in {data_dir} with multi-source feature columns."
        )

    df = pd.read_csv(csv_path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

    input_features = config.get("input_features", {}).get(building)
    if not input_features:
        raise ValueError(
            f"No input_features defined for building '{building}' in {args.config}. "
            f"Add input_features.{building} with a list of column names."
        )

    missing = [c for c in input_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {csv_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    output_features = ["Power (kW)"]
    if output_features[0] not in df.columns:
        raise ValueError(f"Target column 'Power (kW)' not found in {csv_path}")

    if "Hour" in df.columns and "Hour" in input_features:
        df["Hour"] = np.cos(df["Hour"].astype(float) * (2 * np.pi / 23))

    required_cols = input_features + output_features
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    n = len(df)
    train_end = int(0.7 * n)
    val_end = train_end + int(0.1 * n)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    train_x = scaler_x.fit_transform(np.array(train_df[input_features]))
    val_x = scaler_x.transform(np.array(val_df[input_features]))
    test_x = scaler_x.transform(np.array(test_df[input_features]))

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y = scaler_y.fit_transform(np.array(train_df[output_features]))
    val_y = scaler_y.transform(np.array(val_df[output_features]))
    test_y = scaler_y.transform(np.array(test_df[output_features]))

    window_size = config["training"]["window_size"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience", 10)

    input_dim = len(input_features)
    output_dim = 1

    model_types = config["models"]["model_types"]
    checkpoint_dir = os.path.join(base_dir, args.checkpoint_dir, exp_name)
    out_dir = os.path.join(base_dir, args.out_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"EXP 2 Multivariate: building={building}, input_dim={input_dim}, features={input_features[:5]}...")

    all_rows = []
    for days in args.horizons:
        horizon_steps = int(days) * 24
        dl_train, _, _ = create_windowed_loader(window_size, horizon_steps, batch_size, train_x, train_y)
        dl_val, X_val, y_val = create_windowed_loader(window_size, horizon_steps, batch_size, val_x, val_y)
        _, X_test, y_test = create_windowed_loader(window_size, horizon_steps, batch_size, test_x, test_y)

        for model_type in model_types:
            train_time_sec = 0.0
            if model_type != PERSISTENCE_MODEL:
                trainer = ModelTrainer(
                    device=device,
                    patience=patience,
                    checkpoint_dir=checkpoint_dir,
                    verbose=args.verbose,
                    log_every=args.log_every,
                    config=config,
                )
                _, train_time_sec = trainer.train(
                    dl_train, dl_val, building, epochs, model_type,
                    input_dim, output_dim, window_size, horizon_steps
                )

            tester = ModelTester(
                input_dim, window_size, output_dim, building,
                device=device, checkpoint_dir=checkpoint_dir,
                model_config=config.get("models", {}),
            )
            y_val_pred, y_val_true = tester.test(X_val, y_val, horizon_steps, model_type, scaler_y)
            y_test_pred, y_test_true = tester.test(X_test, y_test, horizon_steps, model_type, scaler_y)

            val_mae, val_rmse, val_r2, val_mape = compute_metrics(y_val_true.flatten(), y_val_pred.flatten())
            test_mae, test_rmse, test_r2, test_mape = compute_metrics(y_test_true.flatten(), y_test_pred.flatten())

            val_wmape = compute_weighted_mape(y_val_true, y_val_pred, horizon_steps)
            test_wmape = compute_weighted_mape(y_test_true, y_test_pred, horizon_steps)

            if args.verbose:
                print(
                    f"[eval] building={building} model={model_type} horizon_days={days} horizon_steps={horizon_steps} "
                    f"val(MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R2={val_r2:.4f}, MAPE={val_mape:.4f}, wMAPE={val_wmape:.4f}) "
                    f"test(MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}, MAPE={test_mape:.4f}, wMAPE={test_wmape:.4f})"
                )

            all_rows.append({
                "building": building,
                "model": model_type,
                "horizon_days": days,
                "horizon_steps": horizon_steps,
                "split": "val",
                "MAE": val_mae,
                "RMSE": val_rmse,
                "R2": val_r2,
                "MAPE": val_mape,
                "wMAPE": val_wmape,
                "train_time_sec": train_time_sec,
            })
            all_rows.append({
                "building": building,
                "model": model_type,
                "horizon_days": days,
                "horizon_steps": horizon_steps,
                "split": "test",
                "MAE": test_mae,
                "RMSE": test_rmse,
                "R2": test_r2,
                "MAPE": test_mape,
                "wMAPE": test_wmape,
                "train_time_sec": train_time_sec,
            })

    df_out = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, f"{building}_scores.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
