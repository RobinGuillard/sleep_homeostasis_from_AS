import importlib.resources
from collections import defaultdict
from typing import Any, Sequence, overload

import numpy as np
import pandas as pd
import torch
import yaml

from sleep.unified_for_PVT import Unified_for_PVT
from sleep.unified import Unified
from sleep.two_process import TwoProcess
from sleep.utils.io import normalise_inputs, to_flat_cpu_array

with importlib.resources.files("sleep").joinpath("params.yaml").open("r") as f:
    PARAMS = yaml.safe_load(f)
import matplotlib.pyplot as plt

METHODS = {
    "unified": Unified,
    "two_process": TwoProcess,
    "unified_for_PVT": Unified_for_PVT,  # same backend as unified but with PVT output
}
SUPPORTED = list(METHODS.keys())


def array_to_diary(
    arr: np.ndarray,
    key: str = "participant",
    rested_idx: int = 0,
) -> pd.DataFrame:
    """Convert a (2, N) sleep/wake array to a diary DataFrame.

    Parameters
    ----------
    arr : np.ndarray, shape (2, N)
        Row 0: time axis (in hours).
        Row 1: sleep/wake signal (0 = wake, 1 = sleep).
    key : str
        Identifier for this participant/diary.
    rested_idx : int
        Index (0-based) in the resulting diary rows considered as the
        fully-rested reference point. Index 0 is the initial row
        (asleep=NaN), index 1 is the first sleep bout, etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``key``, ``asleep``, ``awake``, ``rested``
        compatible with the ``Model`` class.
    """
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError("arr must have shape (2, N): row 0 = time, row 1 = sleep/wake signal.")

    time = arr[0]
    signal = arr[1].astype(int)

    if not np.all(np.isin(signal, [0, 1])):
        raise ValueError("Signal row must contain only 0 (wake) and 1 (sleep).")

    # Extract transitions (0=wake, 1=sleep)
    diffs = np.diff(signal)
    asleep_times = time[np.where(diffs == 1)[0] + 1]   # wake→sleep (0→1)
    awake_times  = time[np.where(diffs == -1)[0] + 1]  # sleep→wake (1→0)

    # Handle boundary: recording starts during sleep
    if signal[0] == 1:
        asleep_times = np.concatenate([[time[0]], asleep_times])
    # Handle boundary: recording ends during sleep
    if signal[-1] == 1:
        awake_times = np.concatenate([awake_times, [time[-1]]])

    n_bouts = min(len(asleep_times), len(awake_times))
    if n_bouts == 0:
        raise ValueError("No complete sleep bouts found in the signal.")

    asleep_times = asleep_times[:n_bouts]
    awake_times  = awake_times[:n_bouts]

    # Build diary rows. Model convention:
    #   row 0      : asleep=NaN, awake=time[0]  (start of recording)
    #   row i>=1   : asleep=asleep_times[i-1], awake=awake_times[i-1]
    # Each row corresponds to exactly one sleep bout (correct 1-to-1 pairing).
    rows = [{"key": key, "asleep": np.nan, "awake": time[0], "rested": False}]
    for i in range(n_bouts):
        rows.append({
            "key": key,
            "asleep": asleep_times[i],
            "awake": awake_times[i],
            "rested": False,
        })

    df = pd.DataFrame(rows)

    if rested_idx < 0 or rested_idx >= len(df):
        raise ValueError(
            f"rested_idx={rested_idx} is out of range for {len(df)} diary rows "
            f"(after filtering bouts < {min_sleep_duration}h)."
        )
    df.loc[rested_idx, "rested"] = True

    return df

KEY_TYPE = str | int | Sequence | np.ndarray | pd.Series
TIME_TYPE = float | Sequence | np.ndarray | pd.Series


class Model:
    """Sleep pressure model.

    This class provides an interface to compute and plot sleep pressure.
    It supports various input types (scalars, arrays, tensors) and
    automatically handles GPU acceleration when available.

    Parameters
    ----------
    name : str
        Name of the backend model to use (e.g., "unified").
    diary : pd.DataFrame
        DataFrame containing sleep/wake transitions with columns:
        - "key": Diary identifier
        - "awake": Timestamps of sleep to wake transitions
        - "asleep": Timestamps of wake to sleep transitions

    Example
    -------
    >>> model = Model("unified", diary)
    >>> out = model.compute(key, time)

    Returns
    -------
    dict[str, torch.Tensor | np.ndarray]
        Dictionary containing sleep metrics with shapes:
        - Scalar inputs: (1,)
        - Vector inputs: (n,) or (n, p) where:
          - n: number of subjects
          - p: number of parameter combinations
        - Matrix inputs: (n, p, m) where:
          - n: number of subjects
          - p: number of parameter combinations
          - m: number of time points

        The return type depends on the input type:
        - If time is torch.Tensor: returns dict[str, torch.Tensor]
        - Otherwise: returns dict[str, np.ndarray]
    """

    @classmethod
    def from_array(
        cls,
        name: str,
        arr: np.ndarray,
        rested_idx: int,
        key: str = "participant",
    ) -> "Model":
        """Instantiate a Model from a (2, N) sleep/wake NumPy array.

        Parameters
        ----------
        name : str
            Backend model name (e.g. ``"unified_for_PVT"``).
        arr : np.ndarray, shape (2, N)
            Row 0: time axis in hours.
            Row 1: binary sleep/wake signal (0 = wake, 1 = sleep).
        rested_idx : int
            0-based index of the diary row that represents the fully-rested
            reference point (passed directly to :func:`array_to_diary`).
            Index 0 is the initial row (asleep=NaN), index 1 is the first
            sleep bout, etc.
        key : str, optional
            Label used as the diary identifier. Defaults to ``"participant"``.

        Returns
        -------
        Model
            A fully initialised :class:`Model` instance.

        Example
        -------
        >>> arr = np.load("final_sleep_wake.npy")   # shape (2, N)
        >>> model = Model.from_array("unified_for_PVT", arr, rested_idx=5)
        >>> out = model.compute("participant", time_array)
        """
        diary = array_to_diary(arr, key=key, rested_idx=rested_idx)
        return cls(name, diary)

    def __init__(self, name: str, diary: pd.DataFrame):
        """Initialize the Model class."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.diary = diary
        if name not in SUPPORTED:
            raise ValueError(f"Model {name} is not supported. Supported models: {SUPPORTED}")
        self.backend = METHODS[name]()

        self.key_id = {k: i for i, k in enumerate(diary["key"].unique())}
        self.id_sc = self._map_id_to_state_change(diary)
        self.defaults = PARAMS[name]["default"]

    def _map_id_to_state_change(self, df):
        """Map IDs to state changes."""
        id_sc = {}
        for k, g in df.groupby("key"):
            sc = pd.concat([g.awake, g.asleep, pd.Series([np.inf])]).dropna().to_numpy()
            sc.sort()
            if g.rested.sum() != 1:
                raise ValueError(f"Diary {k} must have exactly one fully rested point.")
            rested = np.argwhere(sc == g.loc[g.rested, "awake"].item()).flatten()[0]
            id_sc[self.key_id[k]] = (
                torch.as_tensor(sc, device=self.device, dtype=torch.float32),
                torch.as_tensor(rested, device=self.device, dtype=torch.int64),
            )
        return id_sc

    def __str__(self) -> str:
        """Return a user-friendly string representation of the model."""
        summary = f"Total of {len(self.key_id)} diaries provided."
        summary += "\n" + 40 * "-"
        summary += f"\n{self.backend}"
        return summary

    @overload
    def compute(
        self, key: torch.Tensor, time: torch.Tensor, **params: Any
    ) -> dict[str, torch.Tensor]: ...

    @overload
    def compute(self, key: KEY_TYPE, time: TIME_TYPE, **params: Any) -> dict[str, np.ndarray]: ...

    def compute(
        self, key: KEY_TYPE | torch.Tensor, time: TIME_TYPE | torch.Tensor, **params: Any
    ) -> dict[str, torch.Tensor] | dict[str, np.ndarray]:
        """Compute the model output."""
        # Whether to return tensors or numpy arrays
        use_tensor = isinstance(key, torch.Tensor) or isinstance(time, torch.Tensor)

        # Merge default parameters with user-provided parameters
        unknown = set(params.keys()).difference(self.defaults.keys())
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")
        merged = {**self.defaults, **params}

        # Normalise inputs
        ids, time, params = normalise_inputs(
            key, time, merged, self.key_id, self.device, torch.float32
        )
        self.backend.check_params(**params)
        n, m = ids.numel(), time.shape[1]
        p = params[list(params.keys())[0]].shape[1]

        # Collapse duplicate ids to avoid computing multiple times the same diary
        uniq_ids, inv = torch.unique(ids, dim=0, return_inverse=True)
        idx_uid_rows: dict[int, list[int]] = defaultdict(list)
        for idx_row, idx_uid in enumerate(inv.tolist()):
            idx_uid_rows[idx_uid].append(idx_row)

        # outputs per original row (keep grad via views)
        out_rows: dict[str, list] = {o: [None] * n for o in self.backend.outputs}

        # Compute once per unique diary
        for idx_uid, idx_rows in idx_uid_rows.items():
            sc, rested = self.id_sc[uniq_ids[idx_uid].item()]

            # stack parameters of the same diary → (1,pi)
            params_uid = {k: v[idx_rows].reshape(1, -1) for k, v in params.items()}
            time_uid = time[idx_rows].flatten()
            out_blk = self.backend.compute(sc, rested, time_uid, **params_uid)

            # slice back - views preserve autograd
            for j, i in enumerate(idx_rows):
                slp0, slp1 = j * p, (j + 1) * p
                slm0, slm1 = j * m, (j + 1) * m
                for k, v in out_blk.items():
                    out_rows[k][i] = v[slp0:slp1, slm0:slm1]

        # Stack all rows to (n,p,m)
        out = {k: torch.stack(v, dim=0).squeeze() for k, v in out_rows.items()}

        if not use_tensor:  # convert to float
            return {k: v.detach().cpu().numpy() for k, v in out.items()}

        return out

    def plot(self, key: KEY_TYPE, step: int = 10, **params: Any) -> tuple[plt.Figure, np.ndarray]:
        """Plot the model output."""
        params = {**self.defaults, **params}
        key = to_flat_cpu_array(key)

        # Find m, the maximum number of points in the diaries
        n, m = len(key), 0
        grouped = self.diary.set_index("key").loc[key].groupby("key", sort=False)
        for k, g in grouped:
            length_hours = g["awake"].max() - g["awake"].min()
            m = max(m, 1 + int(length_hours * 60) // step)  # number of points in the diary

        # Create time array
        time = np.full((n, m), np.nan)
        mi = []  # number of points in the diary
        for i, k in enumerate(key):
            g = grouped.get_group(k)
            ti = np.arange(g["awake"].min() * 60, g["awake"].max() * 60, step)
            time[i, : len(ti)] = ti / 60  # in hours
            mi.append(len(ti))

        # Compute the model output
        out = self.compute(key, time, **params)
        out = {k: v.reshape(n, -1, m) for k, v in out.items()}
        p = out[self.backend.outputs[0]].shape[1]

        if n > 3:
            raise ValueError(f"Too many key to plot. Please use {n} <= 3.")
        if p > 3:
            raise ValueError(f"Too many different parameters to plot. Please use {p} <= 3.")

        # Prepare the panels' titles
        npanels = n * p
        params_diff = [["" for _ in range(n)] for _ in range(p)]
        params_comm = ""
        for k, v in params.items():
            if isinstance(v, float | int):
                params_comm += f"{k}={v}, "
            else:
                v = np.asarray(v).flatten()
                if len(v) == n > 1:
                    for i in range(n):
                        for j in range(p):
                            params_diff[j][i] += f"{k}={v[i]}, "
                elif len(v) == p > 1:
                    for i in range(n):
                        for j in range(p):
                            params_diff[j][i] += f"{k}={v[j]}, "
                elif len(v) > p or len(v) > p:
                    for ij in range(npanels):
                        params_diff[ij % n][ij // n] += f"{k}={v[ij]}, "
                else:
                    params_comm += f"{k}={v[0]}, "

        # Figure layout
        width_ratios = [mi[i] for i in range(n)]
        fig, axs = plt.subplots(
            p,
            n,
            figsize=(12, p * 3),
            sharey=True,
            sharex="col",
            gridspec_kw={"width_ratios": width_ratios},
        )
        axs = np.array(axs).reshape(p, n)

        for i in range(n):
            g = grouped.get_group(key[i])
            for j in range(p):
                ax = axs[j][i]
                self.backend.plot(
                    ax, g, time[i, : mi[i]], **{k: v[i, j, : mi[i]] for k, v in out.items()}
                )
                if (i == 0 and p in (1, 2)) or (i == 0 and p == 3 and j == 1):
                    ax.set_ylabel("Impairment")
                if (j == p - 1 and n in (1, 2)) or (j == p - 1 and n == 3 and i == 1):
                    ax.set_xlabel("Time (days)")

                title = ""
                if j == 0:
                    title = f"{key[i]}".capitalize()
                if params_diff[j][i]:
                    if title != "":  # new line if there is already a title
                        title += "\n"
                    title += f"{params_diff[j][i][:-2]}"
                ax.set_title(title)

                # extract the legend
                handles, labels = ax.get_legend_handles_labels()

        axs[0, 0].legend(handles, labels, loc="upper right")

        # deactivate the unused axes
        for ax in axs[npanels:]:
            ax.set_visible(False)

        fig.suptitle(f"{self.name.capitalize()}: {params_comm[:-2]}")
        plt.tight_layout()
        plt.show()

        return fig, axs
