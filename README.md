# Multi-horizon Building Load Forecasting with LSTM-T-KAN

Official implementation for the paper *Multi-horizon Building Load Forecasting with an LSTM Encoder and a Temporal Kolmogorov–Arnold Network*.

## Experiments


| Experiment | Input                                |
| ---------- | ------------------------------------ |
| **EXP 1**  | Univariate (historical load only)    |
| **EXP 2**  | Multivariate (multi-source features) |


### EXP 1: Univariate Baseline

- **Input**: Historical load (`Power (kW)`) only
- **Models**: Persistence, MLP, GRU, LSTM, KAN, LSTM-T-KAN, LSTM-MLP, TCN, N-BEATS, Transformer, CatBoost, XGBoost
- **Horizons**: 1 to 30 days

### EXP 2: Multivariate

- **Input**: Multi-source features (calendar, load statistics, time-frequency, lagged load). See `configs/exp2.yaml` for the full feature list.
- **Models**: Same as EXP 1
- **Horizons**:  1 to 30  days
- **Data**: Requires `office.csv` with pre-engineered columns (e.g. `Near_base_load`, `P_t_1`, `Hour`, `Is_holiday`, etc.)

## Project Structure

```
├── configs/
│   ├── exp1.yaml              # EXP 1: univariate config
│   └── exp2.yaml              # EXP 2: multivariate config (input_features)
├── data/raw/                  # Raw data (office.csv with multi-source features)
├── scripts/
│   ├── run_exp1.py            # EXP 1: univariate train & evaluate
│   └── run_exp2.py            # EXP 2: multivariate train & evaluate
├── src/load_forecasting/
│   ├── models/                # Model definitions
│   ├── data/                  # Windowing, preprocessing
│   ├── training/              # Trainer
│   ├── evaluation/            # Tester, metrics
│   └── utils/                 # Reproducibility
├── checkpoints/
│   ├── exp1/                  # Saved weights (EXP 1)
│   └── exp2/                  # Saved weights (EXP 2)
├── results/
│   ├── exp1/                  # Evaluation CSV (EXP 1)
│   └── exp2/                  # Evaluation CSV (EXP 2)
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

## Quick Start

```bash
# 1. Place office.csv in data/raw/

# 2. Run experiments
python scripts/run_exp1.py   # Univariate
python scripts/run_exp2.py   # Multivariate
```

## Usage

**EXP 1 :**

```bash
python scripts/run_exp1.py
python scripts/run_exp1.py --horizons 5 10 15 20 25 30
python scripts/run_exp1.py --verbose --device cpu
```

**EXP 2 :**

```bash
python scripts/run_exp2.py
python scripts/run_exp2.py --building office --horizons 5 10 15 20 25 30
python scripts/run_exp2.py --verbose --device cpu
```

## Output

- **Checkpoints**: `checkpoints/{exp1|exp2}/{building}_{model}_len{steps}.pt` (neural) or `.pkl` (CatBoost, XGBoost)
- **Results**: `results/{exp1|exp2}/{building}_scores.csv` (MAE, RMSE, R², MAPE, wMAPE, time)

## Configuration

- **EXP 1**: `configs/exp1.yaml` — `model_types`, hyperparameters, training settings
- **EXP 2**: `configs/exp2.yaml` — `input_features` per building, model config, training settings

## License

MIT License. See [LICENSE](LICENSE) for details.