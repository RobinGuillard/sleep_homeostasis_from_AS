# Sleep

This package provides tools for analyzing and simulating sleep pressure dynamics from sleep diaries. The computation is done using PyTorch for differentiability and GPU support, making the models suitable for parameter fitting via gradient-based optimization.

Currently supported models:

| Model name | Class | Description |
|---|---|---|
| `unified` | `Unified` | Unified sleep model (homeostasis + debt) |
| `unified_for_PVT` | `Unified_for_PVT` | Unified model extended with PVT and circadian output |
| `two_process` | `TwoProcess` | Classic two-process model |
| `REM_three_process_charge` | `REMThreeProcessCharge` | Three-state REM model — wake charges homeostasis |
| `REM_three_process_discharge` | `REMThreeProcessDischarge` | Three-state REM model — wake discharges homeostasis |
| `REM_three_process_neutral` | `REMThreeProcessNeutral` | Three-state REM model — wake has no effect on homeostasis |

---

## Table of Contents

- [Installation](#installation)
- [Getting started](#getting-started)
  - [Sleep diary format](#sleep-diary-format)
  - [Loading from a numpy array](#loading-from-a-numpy-array)
  - [Example](#example)
- [Models](#models)
  - [Unified](#unified)
  - [Unified for PVT](#unified-for-pvt)
  - [REM three-process models](#rem-three-process-models)
- [References](#references)

---

## Installation

To install **sleep** directly from Git:

```bash
pip install git+https://github.com/AdrienSpecht/sleep.git@main
```

Alternatively, with Poetry:

```bash
poetry add git+https://github.com/AdrienSpecht/sleep.git@main
```

---

## Getting started

### Sleep diary format

Sleep diaries should be provided as a `pd.DataFrame` with the following columns:

| Column | Type | Description |
|---|---|---|
| `key` | str | Identifier for the sleep schedule (e.g. `"Control: 16h / 8h"`) |
| `asleep` | float | Time in hours when sleep begins (`NaN` for the initial row) |
| `awake` | float | Time in hours when sleep ends |
| `rested` | bool | `True` for exactly one row per key, marking the fully-rested reference point |

Example:

| key | asleep | awake | rested |
|---|---|---|---|
| Control: 16h / 8h | | 0.0 | True |
| Control: 16h / 8h | 16.0 | 24.0 | |
| Control: 16h / 8h | 40.0 | 48.0 | |

- The first row (empty `asleep`, `awake=0.0`) marks the start of the recording. The `rested=True` flag on this row means the participant is assumed fully rested at time 0.
- Each subsequent row represents one sleep bout: `asleep` is the sleep onset, `awake` is the wake time, both in hours.
- Multiple schedules can be combined in the same DataFrame, distinguished by their `key`.

### Loading from a numpy array

If your sleep/wake data is stored as a continuous binary signal (e.g. from actigraphy or PSG scoring), you can instantiate a `Model` directly from a `(2, N)` NumPy array using the `from_array` classmethod, without manually building a diary.

**Array format:**
- Row 0: time axis in hours
- Row 1: binary sleep/wake signal — `0 = wake`, `1 = sleep`

```python
import numpy as np
from sleep import Model

arr = np.load("sleep_wake.npy")   # shape (2, N)

# rested_idx: 0-based index of the diary row considered as the fully-rested
# reference point. Index 0 = the initial recording-start row (asleep=NaN).
# Index k>=1 = the k-th sleep bout.
model = Model.from_array("unified_for_PVT", arr, rested_idx=5, key="participant")

time = np.linspace(arr[0, 0], arr[0, -1], 1000)   # hours
out = model.compute("participant", time)
```

The conversion (`array_to_diary`) detects all sleep/wake transitions in the signal and builds a diary row per sleep bout. All micro-transitions are preserved — no filtering is applied.

### Example

```python
import pandas as pd
from sleep import Model

diary = pd.read_csv("data/diary.csv")

model = Model("unified", diary)

# Compute for a single schedule and specific time points
out = model.compute(key="Control: 16h / 8h", time=[10, 15, 50])

# Plot all schedules
fig, axs = model.plot(key=diary.key.unique())
```

---

## Models

### Unified

**Name:** `"unified"` — **Outputs:** `homeostasis`, `debt`

A two-variable model of sleep-wake dynamics tracking homeostatic sleep pressure `s` and sleep debt `d` [1].

**Differential equations:**

During wake:
```
ds/dt = (1 - s) / t_w
dd/dt = (-d + 1) / t_la
```

During sleep:
```
ds/dt = -(s - d) / t_s
dd/dt = (-d - wsr) / t_la
```

where `wsr = (T - need) / need` is the wake-sleep ratio.

**Parameters:**

| Parameter | Description |
|---|---|
| `t_w` | Time constant during wake |
| `t_s` | Time constant during sleep |
| `t_la` | Time constant for debt dynamics |
| `need` | Sleep need (hours) |
| `T` | Period (hours) |

**Constraints:** `t_s < t_la`, `t_w < t_la`

---

### Unified for PVT

**Name:** `"unified_for_PVT"` — **Outputs:** `homeostasis`, `debt`, `PVT`, `circadian`

Extends the Unified model with a predicted psychomotor vigilance test (PVT) score, computed as the sum of homeostasis and a scaled circadian term:

```
PVT(t) = s(t) + kappa * C(t)
```

where `C(t)` is a five-harmonic circadian oscillator [1]:

```
C(t) = 0.97·sin(2π·t/24) + 0.22·sin(4π·t/24) + 0.07·sin(6π·t/24)
      + 0.03·sin(8π·t/24) + 0.001·sin(10π·t/24)

with t = (time + phi + initial_hour) mod 24
```

**Additional parameters:**

| Parameter | Description |
|---|---|
| `U` | Scaling factor for homeostasis asymptote |
| `kappa` | Sensitivity to the circadian rhythm |
| `phi` | Phase of the circadian rhythm (hours) |
| `initial_hour` | Clock hour at `time=0` (default: 8h) |

---

### REM three-process models

**Names:** `"REM_three_process_charge"`, `"REM_three_process_discharge"`, `"REM_three_process_neutral"`  
**Outputs:** `homeostasis`

A family of three-state models that distinguish NREM and REM sleep stages, allowing finer-grained modelling of REM homeostasis. Each model uses the same NREM and REM dynamics but differs in how **wake** affects homeostasis:

| Variant | Wake dynamics |
|---|---|
| `charge` | Wake charges homeostasis: `ds/dt = (1 - s) / t_w` |
| `discharge` | Wake discharges homeostasis: `ds/dt = -s / t_w` |
| `neutral` | Wake has no effect on homeostasis: `ds/dt = 0` |

**NREM sleep** (charges homeostasis):
```
ds/dt = (1 - s) / t_NREM
```

**REM sleep** (discharges homeostasis toward `s_target`, default 0):
```
ds/dt = -(s - s_target) / t_REM
```

Sleep architecture within a night is modelled as `N_cycles` successive NREM–REM cycles. The REM fraction per cycle increases across the night (REM is concentrated toward the end), controlled by `cycle_increment`.

**Parameters:**

| Parameter | Description |
|---|---|
| `t_w` | Time constant during wake |
| `t_NREM` | Time constant during NREM sleep |
| `t_REM` | Time constant during REM sleep |
| `need` | Total sleep need (hours) |
| `T` | Period (hours) |
| `REM_relative_need` | Fraction of sleep need that is REM (in `(0, 1)`) |
| `cycle_increment` | Controls the gradient of REM concentration across cycles (`[0, 1]`) |

**Input format:** these models require a hypnogram signal with three states: `0 = WAKE`, `1 = NREM`, `4 = REM`. They are used via a dedicated `ModelHypnogram` interface rather than the standard `Model` class.

---

## References

[1] P. Rajdev et al., "A unified mathematical model to quantify performance impairment for both chronic sleep restriction and total sleep deprivation.", *Journal of Theoretical Biology*, 2013.