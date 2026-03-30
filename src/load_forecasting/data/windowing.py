import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_windowed_loader(
    window: int,
    horizon_steps: int,
    batch_size: int,
    data_x: np.ndarray,
    data_y: np.ndarray,
):
    """
    Convert a continuous series into sliding-window samples and build a DataLoader.

    Inputs:
      data_x: [N, F]
      data_y: [N, 1] (univariate target only)
    Outputs:
      dataloader: (x, y) where x=[B, window, F], y=[B, horizon_steps]
    """
    input_seq_len = window
    total_seq_len = input_seq_len + horizon_steps

    def sliding_window(arr, win_size):
        shape = (arr.shape[0] - win_size + 1, win_size) + arr.shape[1:]
        strides = (arr.strides[0],) + arr.strides
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    n_samples_x = data_x.shape[0]
    n_samples_y = data_y.shape[0]
    if data_y.ndim != 2 or data_y.shape[1] != 1:
        raise ValueError(
            f"create_windowed_loader expects univariate targets with shape [N, 1]; got {data_y.shape}."
        )
    if n_samples_x < total_seq_len or n_samples_y < total_seq_len:
        raise ValueError(
            f"Insufficient data length: need >= {total_seq_len}, got X={n_samples_x}, Y={n_samples_y}."
        )

    result_x = sliding_window(data_x, total_seq_len)
    result_y = sliding_window(data_y, total_seq_len)

    x_data = result_x[:, :input_seq_len]
    y_data = result_y[:, -horizon_steps:, 0]

    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], -1))
    y_data = np.reshape(y_data, (y_data.shape[0], -1))

    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, x_tensor, y_tensor
