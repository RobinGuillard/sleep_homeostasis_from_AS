
from __future__ import annotations

import importlib.resources
from collections import defaultdict
from typing import Any, Sequence, overload
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import torch
import yaml


from sleep.REM_three_process_charge import REMThreeProcessCharge
from sleep.utils.io import normalise_inputs, to_flat_cpu_array

with importlib.resources.files("sleep").joinpath("params.yaml").open("r") as f:
    PARAMS = yaml.safe_load(f)
import matplotlib.pyplot as plt

METHODS = {
    "rem_three_process": REMThreeProcessCharge    
}

"""
model_hypnogram.py
==================
Nouvelle version de la classe Model qui accepte en entrée soit :
  - un diary DataFrame classique (format existant, pour les modèles two-process / unified)
  - un hypnogramme NumPy (format nouveau, pour REMThreeProcessCharge)

Et version révisée de REMThreeProcessCharge dont compute() et infer_initial_state()
utilisent la représentation à trois états (wake / NREM / REM) issue de l'hypnogramme.
"""


# ---------------------------------------------------------------------------
# Helpers / types
# ---------------------------------------------------------------------------
KEY_TYPE = str | int | Sequence | np.ndarray | pd.Series
TIME_TYPE = float | Sequence | np.ndarray | pd.Series

WAKE  = 0
NREM  = 1   # collapsed (epochs 1, 2, 3 → NREM)
REM   = 4


# ---------------------------------------------------------------------------
#  Conversion hypnogramme -> transitions d'état
# ---------------------------------------------------------------------------

def hypnogram_to_state_changes(
    hypnogram: np.ndarray,
    epoch_duration_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convertit un hypnogramme en séquence de transitions d'état.

    Les stades N1/N2/N3 (valeurs 1, 2, 3) sont regroupés en NREM (1).
    Wake reste 0, REM reste 4.

    Parameters
    ----------
    hypnogram : np.ndarray, shape (n_epochs,)
        Entiers 0-4 : 0=wake, 1/2/3=NREM, 4=REM.
    epoch_duration_s : float
        Durée d'une époque en secondes (défaut 30 s).

    Returns
    -------
    sc_times  : np.ndarray, shape (n_transitions,)
        Timestamp (heures) du début de chaque segment homogène.
    sc_states : np.ndarray, shape (n_transitions,)
        État {WAKE=0, NREM=1, REM=4} de chaque segment.
    """
    collapsed = np.where(hypnogram == 0, WAKE,
                np.where(hypnogram == 4, REM, NREM)).astype(np.int64)

    change_idx = np.concatenate([[0], np.where(np.diff(collapsed) != 0)[0] + 1])

    sc_times  = change_idx * epoch_duration_s / 3600.0
    sc_states = collapsed[change_idx]
    return sc_times, sc_states



# ---------------------------------------------------------------------------
#  ModelHypnogram
# ---------------------------------------------------------------------------

class ModelHypnogramREM:
    """Calcule l'homéostasie REM à partir d'un hypnogramme NumPy.

    Parameters
    ----------
    backend : REMThreeProcessCharge
        Instance du modèle à trois états.
    hypnogram : np.ndarray, shape (n_epochs,)
        Entiers 0-4 (0=wake, 1/2/3=NREM, 4=REM).
    epoch_duration_s : float
        Durée d'une époque en secondes (défaut 30 s).
    rested_epoch : int
        Indice de l'époque désignée comme reposée (point de calibration).
        Ignoré en mode "free".
    N_cycles : int
        Nombre de cycles NREM/REM dans la journée idéale (défaut 5).
    s0_mode : {"stable", "free"}
        "stable" : s0 calculé depuis _periodic_value + backstep (défaut).
        "free"   : s0 est un paramètre libre optimisable, initialisé à s0_init.
                   Permet un gradient correct pour TOUS les paramètres.
    s0_init : float
        Valeur initiale de s0 en mode "free" (défaut 0.5).
    default_params : dict, optional
        Valeurs par défaut des paramètres du modèle.

    Example
    -------
    >>> hyp   = np.load("aggregated_hypnogram.npy")
    >>> # Mode stable (défaut)
    >>> model = ModelHypnogramREM(REMThreeProcessCharge(), hyp,
    ...                        epoch_duration_s=30, rested_epoch=4030)
    >>> # Mode free
    >>> model = ModelHypnogramREM(REMThreeProcessCharge(), hyp,
    ...                        epoch_duration_s=30, rested_epoch=0,
    ...                        s0_mode="free", s0_init=0.3)
    >>> out = model.compute(np.arange(0, 84, 1/6))
    """

    def __init__(
        self,
        backend:          REMThreeProcessCharge,
        hypnogram:        np.ndarray,
        *,
        epoch_duration_s: float = 30.0,
        rested_epoch:     int,
        N_cycles:         int = 5,
        s0_mode:          str = "stable",
        s0_init:          float = 0.5,
        default_params:   dict[str, float] | None = None,
    ):
        if s0_mode not in ("stable", "free"):
            raise ValueError("s0_mode doit être 'stable' ou 'free'")

        self.backend  = backend
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N_cycles = N_cycles
        self.s0_mode  = s0_mode

        sc_times_np, sc_states_np = hypnogram_to_state_changes(hypnogram, epoch_duration_s)
        print(sc_times_np)
        print(sc_states_np)

        rested_time_h = rested_epoch * epoch_duration_s / 3600.0
        print(f"rested_time_h : {rested_time_h:.2f} h")
        rested_idx = int(np.searchsorted(sc_times_np, rested_time_h, side="right")) - 1
        print(f"rested_idx : {rested_idx} / {len(sc_times_np)}")
        rested_idx = max(0, rested_idx)

        self.sc_times   = torch.as_tensor(sc_times_np,  device=self.device, dtype=torch.float32)
        self.sc_states  = torch.as_tensor(sc_states_np, device=self.device, dtype=torch.int64)
        self.rested_idx = rested_idx

        self.t_start_h = float(sc_times_np[0])
        self.t_end_h   = float(sc_times_np[-1]) + epoch_duration_s / 3600.0

        # Mode "free" : s0 est un paramètre libre torch.nn.Parameter
        if s0_mode == "free":
            # On paramétrise via logit pour que sigmoid(s0_free) ∈ (0, 1)
            # et le gradient ne soit jamais bloqué par un clamp.
            s0_init_clamped = float(np.clip(s0_init, 1e-4, 1 - 1e-4))
            logit_init = float(np.log(s0_init_clamped / (1 - s0_init_clamped)))
            self.s0_free = torch.nn.Parameter(
                torch.tensor([[logit_init]], dtype=torch.float32, device=self.device)
            )
        else:
            self.s0_free = None

        self.defaults = default_params or {
            "t_w":               16.0,
            "t_NREM":             4.0,
            "t_REM":              1.0,
            "need":               8.0,
            "T":                 24.0,
            "REM_relative_need":  0.25,
            "cycle_increment":    0.8,
        }

    # ------------------------------------------------------------------
    def _to_tensor(self, val: float | list | np.ndarray) -> torch.Tensor:
        """Scalaire ou tableau -> tenseur (1, p)."""
        arr = np.atleast_1d(np.asarray(val, dtype=np.float32))
        return torch.as_tensor(arr, device=self.device).reshape(1, -1)

    # ------------------------------------------------------------------
    def compute(
        self,
        time: np.ndarray | torch.Tensor,
        **params: float | list | np.ndarray,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Calcule l'homéostasie aux instants `time` (en heures).

        Parameters
        ----------
        time : array-like, shape (m,)
            Instants de sortie en heures. Peut contenir des NaN.
        **params :
            Surcharge des paramètres par défaut.

        Returns
        -------
        dict[str, np.ndarray]  (ou dict[str, torch.Tensor] si time est un Tensor)
        """
        use_tensor = isinstance(time, torch.Tensor)
        merged     = {**self.defaults, **params}

        unknown = set(params.keys()) - set(self.defaults.keys())
        if unknown:
            raise ValueError(f"Paramètres inconnus : {unknown}")
        #conversion en tenseurs uniquement si nécessaire, tester si merged contient des tensors déjà
        p_tensors ={}
        for k, v in merged.items():
            if isinstance(v, torch.Tensor):
                p_tensors[k] = v.to(self.device)
            else:
                p_tensors[k] = self._to_tensor(v)

        self.backend.check_params(**p_tensors)

        time_t = time.to(self.device) if use_tensor else torch.as_tensor(
            np.asarray(time, dtype=np.float32), device=self.device
        )

        # ── Résolution de s0 selon le mode ──────────────────────────────
        if self.s0_mode == "free":
            # s0_free est en espace logit -> sigmoid pour rester dans (0,1)
            # Le gradient traverse sigmoid sans jamais être bloqué par un clamp.
            s0 = torch.sigmoid(self.s0_free)   # (1, 1), requires_grad=True
        else:
            # Mode "stable" : calcul depuis _periodic_value + backstep
            s0 = self.backend.infer_initial_state(
                self.sc_times, self.sc_states, self.rested_idx,
                N_cycles=self.N_cycles, **p_tensors,
            )  # (1, p)

        out = self.backend.compute(
            sc_times          = self.sc_times,
            sc_states         = self.sc_states,
            s0                = s0,
            time              = time_t,
            N_cycles          = self.N_cycles,
            **p_tensors,
        )

        if not use_tensor:
            return {k: v.detach().cpu().numpy() for k, v in out.items()}
        return out

    # ------------------------------------------------------------------
    def plot(
        self,
        step_min: int = 10,
        **params: float | list | np.ndarray,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Trace l'homéostasie sur toute la durée de l'enregistrement.

        Parameters
        ----------
        step_min : int
            Pas d'échantillonnage en minutes (défaut 10 min).
        **params :
            Surcharge des paramètres par défaut.
        """
        merged    = {**self.defaults, **params}
        p_tensors = {k: self._to_tensor(v) for k, v in merged.items()}
        p         = p_tensors["t_w"].shape[1]

        time = np.arange(self.t_start_h * 60, self.t_end_h * 60, step_min) / 60.0
        out  = self.compute(time, **params)

        homeostasis = out["homeostasis"]
        if homeostasis.ndim == 1:
            homeostasis = homeostasis[np.newaxis, :]

        fig, axs = plt.subplots(p, 1, figsize=(14, 3 * p), sharex=True)
        if p == 1:
            axs = [axs]

        sc_times_np  = self.sc_times.cpu().numpy()
        sc_states_np = self.sc_states.cpu().numpy()

        for j, ax in enumerate(axs):
            self.backend.plot(
                ax, sc_times_np, sc_states_np, time, homeostasis[j],
                rested_idx=self.rested_idx if self.s0_mode == "stable" else None,
            )
            ax.set_ylabel("Homéostasie")
            title_parts = []
            for k, v in merged.items():
                arr = np.atleast_1d(v)
                title_parts.append(f"{k}={arr[j] if len(arr) > 1 else arr[0]:.2f}")
            ax.set_title(", ".join(title_parts))

        axs[-1].set_xlabel("Temps (jours)")
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles, labels, loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.show()
        return fig, np.array(axs)

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        dur = self.t_end_h - self.t_start_h
        if self.s0_mode == "free":
            s0_str = f"free (s0={torch.sigmoid(self.s0_free).item():.4f})"
        else:
            s0_str = "stable (via _periodic_value + backstep)"
        return (
            f"ModelHypnogram\n"
            f"  Durée : {dur:.1f} h  |  "
            f"Transitions : {len(self.sc_times)}  |  "
            f"rested_idx : {self.rested_idx}  |  "
            f"N_cycles : {self.N_cycles}\n"
            f"  s0_mode : {s0_str}\n"
            f"  Backend : {self.backend}"
            f"  Paramètres par défaut : {self.defaults}"
        )
