# Multi-horizon Building Load Forecasting with LSTM-T-KAN

Official implementation for the paper *Multi-horizon Building Load Forecasting with an LSTM Encoder and a Temporal Kolmogorov–Arnold Network*.

## Experiments


| Experiment | Input                             |
| ---------- | --------------------------------- |
| **EXP 1**  | Univariate (historical load only) |


### EXP 1: Univariate Baseline

- **Input**: Historical load (`Power (kW)`) only
- **Models**: Persistence, MLP, GRU, LSTM, KAN, LSTM-T-KAN, LSTM-MLP, TCN, N-BEATS, Transformer, CatBoost, XGBoost
- **Horizons**: 5, 10, 15, 20, 25, 30 days

## Project Structure

```
├── configs/
│   └── exp1.yaml              # EXP 1 config
├── data/raw/                  # Raw data (office.csv)
├── scripts/
│   └── run_exp1.py     # EXP 1: train & evaluate
├── src/load_forecasting/
│   ├── models/                # Model definitions
│   ├── data/                  # Windowing, preprocessing
│   ├── training/              # Trainer
│   ├── evaluation/            # Tester, metrics
│   └── utils/                 # Reproducibility
├── checkpoints/exp1/         # Saved weights
├── results/exp1/             # Evaluation CSV
├── pyproject.toml
└── environment.yml
```

## Environment Setup

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate load-forecasting
```

**pip:**

```bash
pip install -e .
# or
pip install -r requirements.txt
```

**RTX 5080/5090 (CUDA 12.8):**

```bash
pip uninstall torch torchvision torchaudio -y
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

## Quick Start (EXP 1)

```bash
# 1. Place office.csv in data/raw/

# 2. Run training and evaluation
python scripts/run_exp1.py
```


## Usage

```bash
# Default: all horizons (5–30 days), building=office
python scripts/run_exp1.py

# Custom horizons
python scripts/run_exp1.py --horizons 5 10 15

# Verbose training logs
python scripts/run_exp1.py --verbose

# Force CPU
python scripts/run_exp1.py --device cpu
```

## Output

- **Checkpoints**: `checkpoints/exp1/{building}_{model}_len{steps}.pt` (neural) or `.pkl` (CatBoost, XGBoost)
- **Results**: `results/exp1/{building}_scores.csv` (MAE, RMSE, R², MAPE, wMAPE)

## Configuration

Edit `configs/exp1.yaml` to change `model_types`, hyperparameters, or training settings.

## License

MIT License. See [LICENSE](LICENSE) for details.