import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, "C:/Users/robin/Documents/Postdoc_Stanford/UMP_computation/sleep/src/")
from sleep.model_hypnogram_REM import ModelHypnogramREM, hypnogram_to_state_changes
from sleep.REM_three_process_charge    import REMThreeProcessCharge,    REM, NREM, WAKE
from sleep.REM_three_process_discharge import REMThreeProcessDischarge
from sleep.REM_three_process_neutral   import REMThreeProcessNeutral

# ── Paramètres fixes (non optimisés) ─────────────────────────────────────────
fixed_params = {
    "t_w":               24.0,
    "t_NREM":            100.0,
    "t_REM":              1.0,
    "need":               8.0,
    "T":                 24.0,
    "REM_relative_need":  0.25,
    "cycle_increment":    0.8,
}

EPOCH_S    = 30
RESTED     = 0
N_CYCLES   = 5
N_STEPS    = 300    # steps par restart (réduit car on fait plusieurs restarts)
N_RESTARTS = 5     # nombre de points de départ aléatoires
LR         = 0.05
T_MIN      = np.log(0.1)
T_MAX      = np.log(1000.0)

base_path = "C:\\Users\\robin\\Documents\\Postdoc_Stanford\\proteomics_sleep_debt_computation\\proteomics\\U_sleep_PSG_pipeline\\aggregated_hypnograms\\"
save_path = "C:/Users/robin/Documents/Postdoc_Stanford/UMP_computation/sleep/data/"
all_aggregated_hypnograms = [
    f for f in os.listdir(base_path)
    if f.startswith("aggregated_hypnogram_") and f.endswith(".npy")
]
print("Aggregated hypnograms found:", all_aggregated_hypnograms)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def pearson_loss(h: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - r(h, y) : minimiser = maximiser la corrélation de Pearson.

    Immunisé contre les décalages de moyenne et d'échelle : un signal
    constant donne r=0 → loss=1, évitant le minimum local trivial.
    """
    h_c = h - h.mean()
    y_c = y - y.mean()
    r   = (h_c * y_c).sum() / (
        torch.sqrt((h_c ** 2).sum() * (y_c ** 2).sum() + eps)
    )
    return 1.0 - r


def draw_random_init(rng: np.random.Generator) -> dict:
    """Tire un point de départ log-uniforme dans les bornes physiologiques."""
    return {
        "log_t_w":    rng.uniform(T_MIN, T_MAX),
        "log_t_NREM": rng.uniform(T_MIN, T_MAX),
        "log_t_REM":  rng.uniform(T_MIN, T_MAX),
        "s0":         rng.uniform(0.05, 0.95),
    }


def grid_warm_start(
    model, obs_t: torch.Tensor, y_obs_t: torch.Tensor, n_per_dim: int = 5
) -> dict:
    """Grille grossière sans gradient pour identifier la meilleure zone de départ.

    Évalue la corrélation sur une grille (t_REM × t_NREM × s0) et retourne
    le point de la grille avec le r le plus élevé.
    Rapide car pas de backward().
    """
    best_r, best_init = -np.inf, None
    t_REM_grid  = np.logspace(-1, 1,   n_per_dim)   # 0.1h – 10h
    t_NREM_grid = np.logspace( 0, 3,   n_per_dim)   # 1h   – 1000h
    s0_grid     = np.linspace(0.1, 0.9, n_per_dim)
    # t_w a peu d'influence sur les modèles Charge/Neutral → on le fixe à 30h
    t_w_fixed   = 30.0

    with torch.no_grad():
        for t_REM in t_REM_grid:
            for t_NREM in t_NREM_grid:
                for s0 in s0_grid:
                    # Injecter s0 dans le paramètre libre
                    s0_c = float(np.clip(s0, 1e-4, 1 - 1e-4))
                    model.s0_free.fill_(np.log(s0_c / (1 - s0_c)))
                    try:
                        h = model.compute(
                            time=obs_t,
                            t_w=t_w_fixed, t_NREM=float(t_NREM), t_REM=float(t_REM),
                        )["homeostasis"].squeeze()
                        r = 1.0 - pearson_loss(h, y_obs_t).item()
                    except Exception:
                        continue
                    if r > best_r:
                        best_r    = r
                        best_init = {
                            "log_t_w":    np.log(t_w_fixed),
                            "log_t_NREM": np.log(t_NREM),
                            "log_t_REM":  np.log(t_REM),
                            "s0":         s0,
                        }
    print(f"    [grid] meilleur r={best_r:.3f}  "
          f"t_REM={np.exp(best_init['log_t_REM']):.3f}  "
          f"t_NREM={np.exp(best_init['log_t_NREM']):.3f}  "
          f"s0={best_init['s0']:.3f}")
    return best_init


def run_one_restart(
    model,
    obs_t:    torch.Tensor,
    y_obs_t:  torch.Tensor,
    init:     dict,
    n_steps:  int,
    lr:       float,
) -> tuple[float, dict, list]:
    """Lance une optimisation Adam depuis `init`, retourne (best_loss, fitted, losses)."""

    log_t_w    = torch.tensor(init["log_t_w"],    dtype=torch.float32, requires_grad=True)
    log_t_NREM = torch.tensor(init["log_t_NREM"], dtype=torch.float32, requires_grad=True)
    log_t_REM  = torch.tensor(init["log_t_REM"],  dtype=torch.float32, requires_grad=True)

    s0_c = float(np.clip(init["s0"], 1e-4, 1 - 1e-4))
    with torch.no_grad():
        model.s0_free.fill_(np.log(s0_c / (1 - s0_c)))

    optim     = torch.optim.Adam([log_t_w, log_t_NREM, log_t_REM, model.s0_free], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps, eta_min=1e-4)

    losses = []
    for step in range(n_steps):
        optim.zero_grad()
        t_w_c    = torch.exp(log_t_w.clamp(T_MIN, T_MAX))
        t_NREM_c = torch.exp(log_t_NREM.clamp(T_MIN, T_MAX))
        t_REM_c  = torch.exp(log_t_REM.clamp(T_MIN, T_MAX))

        h    = model.compute(
            time=obs_t, t_w=t_w_c, t_NREM=t_NREM_c, t_REM=t_REM_c,
        )["homeostasis"].squeeze()
        loss = pearson_loss(h, y_obs_t)
        loss.backward()
        optim.step()
        scheduler.step()
        losses.append(loss.item())

    with torch.no_grad():
        fitted = {
            "t_w":    float(torch.exp(log_t_w.clamp(T_MIN, T_MAX))),
            "t_NREM": float(torch.exp(log_t_NREM.clamp(T_MIN, T_MAX))),
            "t_REM":  float(torch.exp(log_t_REM.clamp(T_MIN, T_MAX))),
            "s0":     float(torch.sigmoid(model.s0_free)),
        }
    return losses[-1], fitted, losses


# ═══════════════════════════════════════════════════════════════════════════════
#  Boucle principale par patient
# ═══════════════════════════════════════════════════════════════════════════════

for hyp_file in all_aggregated_hypnograms:
    print(f"\n{'#'*60}\n  Patient : {hyp_file}\n{'#'*60}")
    hyp = np.load(os.path.join(base_path, hyp_file))

    sc_times_np, sc_states_np = hypnogram_to_state_changes(hyp, epoch_duration_s=EPOCH_S)

    # ── Extraction des bouts REM ──────────────────────────────────────────────
    rem_onsets, rem_durations = [], []
    for k in range(len(sc_times_np)):
        if sc_states_np[k] == REM:
            onset = sc_times_np[k]
            end   = sc_times_np[k + 1] if k + 1 < len(sc_times_np) else sc_times_np[-1]
            rem_onsets.append(onset)
            rem_durations.append(end - onset)
    rem_onsets    = np.array(rem_onsets)
    rem_durations = np.array(rem_durations)

    # ── Instanciation des trois modèles ──────────────────────────────────────
    backends = {
        "Neutral":   REMThreeProcessNeutral(),
        "Charge":    REMThreeProcessCharge(),
        "Discharge": REMThreeProcessDischarge(),
    }
    models = {
        name: ModelHypnogramREM(
            backend, hyp,
            epoch_duration_s=EPOCH_S,
            rested_epoch=RESTED,
            N_cycles=N_CYCLES,
            s0_mode="free",
            s0_init=0.3,
            default_params=fixed_params,
        )
        for name, backend in backends.items()
    }

    t_start = models["Charge"].t_start_h
    t_end   = models["Charge"].t_end_h
    n_days  = int((t_end - t_start) / 24)

    # ── Observations : bouts REM ≥ 2 min ─────────────────────────────────────
    mask      = (rem_onsets >= t_start) & (rem_onsets < t_end)
    obs_times = rem_onsets[mask]
    obs_durs  = rem_durations[mask]
    filt      = obs_durs >= 4 * EPOCH_S / 3600
    obs_times = obs_times[filt]
    obs_durs  = obs_durs[filt]

    if len(obs_times) < 5:
        print(f"  Pas assez de bouts REM ({len(obs_times)}), patient ignoré.")
        continue

    print(f"{len(obs_times)} bouts REM ≥ 2 min sur {n_days} jours "
          f"(moy. {len(obs_times)/n_days:.1f}/jour)")

    y_obs   = obs_durs / obs_durs.max()
    obs_t   = torch.as_tensor(obs_times, dtype=torch.float32)
    y_obs_t = torch.as_tensor(y_obs,     dtype=torch.float32)
    t_dense = np.linspace(t_start, t_end, 4000)

    MODEL_STYLE = {
        "Charge":    {"color": "C0"},
        "Discharge": {"color": "tomato"},
        "Neutral":   {"color": "mediumpurple"},
    }

    rng = np.random.default_rng(42)
    results = {}

    # ── Boucle par modèle ─────────────────────────────────────────────────────
    for model_name, model in models.items():
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")

        # Étape 1 : grille grossière pour identifier la meilleure zone
        print("  [1/2] Grille de warm-start...")
        grid_init = grid_warm_start(model, obs_t, y_obs_t, n_per_dim=5)

        # Étape 2 : multi-start
        # Le restart 0 part du meilleur point de la grille,
        # les suivants tirent aléatoirement dans l'espace des paramètres.
        print(f"  [2/2] Multi-start ({N_RESTARTS} restarts × {N_STEPS} steps)...")
        best_loss, best_fitted, best_losses = np.inf, None, None

        for restart in range(N_RESTARTS):
            init = grid_init if restart == 0 else draw_random_init(rng)
            try:
                loss, fitted, losses = run_one_restart(
                    model, obs_t, y_obs_t, init, N_STEPS, LR
                )
            except Exception as e:
                print(f"    restart {restart:>2d}  ERREUR: {e}")
                continue

            r = 1.0 - loss
            print(f"    restart {restart:>2d}  r={r:.3f}  "
                  f"t_REM={fitted['t_REM']:.3f}  t_NREM={fitted['t_NREM']:.3f}  "
                  f"t_w={fitted['t_w']:.3f}  s0={fitted['s0']:.3f}"
                  + ("  ← best" if loss < best_loss else ""))

            if loss < best_loss:
                best_loss   = loss
                best_fitted = fitted
                best_losses = losses

        final_r = 1.0 - best_loss
        print(f"\n  Best r = {final_r:.4f}")
        print(f"  {'Parameter':<10} {'Fitted':>10}")
        for k, v in best_fitted.items():
            print(f"  {k:<10} {v:>10.4f}")

        # Restaurer s0_free au meilleur s0 pour le calcul de la trajectoire dense
        s0_best = float(np.clip(best_fitted["s0"], 1e-4, 1 - 1e-4))
        with torch.no_grad():
            model.s0_free.fill_(np.log(s0_best / (1 - s0_best)))

        s_dense    = model.compute(
            time=t_dense,
            t_w=best_fitted["t_w"],
            t_NREM=best_fitted["t_NREM"],
            t_REM=best_fitted["t_REM"],
        )["homeostasis"].squeeze()
        s_range    = s_dense.max() - s_dense.min()
        s_dense_mm = (s_dense - s_dense.min()) / max(s_range, 1e-6)

        results[model_name] = {
            "losses":     best_losses,
            "fitted":     best_fitted,
            "r":          final_r,
            "s_dense_mm": s_dense_mm,
        }

    # ── Figure (2×2) ──────────────────────────────────────────────────────────
    t_days   = (t_dense   - t_start) / 24
    obs_days = (obs_times - t_start) / 24

    fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                             gridspec_kw={"width_ratios": [1, 3]})

    # Haut-gauche : loss curves (meilleur restart seulement)
    ax_loss = axes[0, 0]
    for name, res in results.items():
        r_val = res["r"]
        ax_loss.plot(
            [1.0 - l for l in res["losses"]],   # afficher r plutôt que loss
            label=f"{name} (r={r_val:.2f})",
            color=MODEL_STYLE[name]["color"],
        )
    ax_loss.set_xlabel("Step (meilleur restart)")
    ax_loss.set_ylabel("Pearson r")
    ax_loss.set_title(f"Convergence (meilleur des {N_RESTARTS} restarts)")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    # Bas-gauche : tableau des paramètres fittés
    ax_tab = axes[1, 0]
    ax_tab.axis("off")
    keys_show  = ["t_w", "t_NREM", "t_REM", "s0", "r"]
    col_labels = ["Param"] + list(results.keys())
    table_data = []
    for k in keys_show:
        row = [k] + [
            f"{res['fitted'][k]:.3f}" if k in res["fitted"] else f"{res['r']:.3f}"
            for res in results.values()
        ]
        table_data.append(row)
    tbl = ax_tab.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.5)
    ax_tab.set_title(f"Best fitted parameters ({N_RESTARTS} restarts)", pad=12)

    # Haut-droit : trajectoires + observations
    ax_traj = axes[0, 1]
    ax_traj.scatter(obs_days, y_obs, color="red", s=14, zorder=4, alpha=0.6,
                    label="REM duration (norm.)")
    for name, res in results.items():
        ax_traj.plot(t_days, res["s_dense_mm"],
                     color=MODEL_STYLE[name]["color"], lw=1.2,
                     label=f"{name} (r={res['r']:.2f})")

    first_rem = True
    for k in range(len(sc_times_np)):
        if sc_states_np[k] != REM:
            continue
        t0  = (sc_times_np[k]    - t_start) / 24
        t1r = sc_times_np[k + 1] if k + 1 < len(sc_times_np) else t_end
        t1  = (t1r - t_start) / 24
        ax_traj.axvspan(t0, t1, color="gold", alpha=0.15,
                        label="REM" if first_rem else "")
        first_rem = False

    ax_traj.set_xlim(0, (t_end - t_start) / 24)
    ax_traj.set_ylim(-0.1, 1.1)
    ax_traj.set_xlabel("Time (days)")
    ax_traj.set_ylabel("Signal [0, 1]")
    ax_traj.set_title(f"Homeostasis (min-max) vs REM duration — {hyp_file}")
    ax_traj.legend(loc="upper right", fontsize=7)
    ax_traj.grid(alpha=0.3)

    # Bas-droit : scatter + corrélation
    ax_scat = axes[1, 1]
    for name, res in results.items():
        s_at_obs = np.interp(obs_times, t_dense, res["s_dense_mm"])
        corr     = np.corrcoef(s_at_obs, y_obs)[0, 1]
        ax_scat.scatter(s_at_obs, y_obs, s=12, alpha=0.5,
                        color=MODEL_STYLE[name]["color"],
                        label=f"{name}  (r={corr:.2f})")

    ax_scat.set_xlabel("Homeostasis (min-max)")
    ax_scat.set_ylabel("REM duration (norm.)")
    ax_scat.set_title("Correlation: fitted homeostasis vs REM duration at onsets")
    ax_scat.legend(fontsize=8)
    ax_scat.grid(alpha=0.3)

    plt.tight_layout()
    out_name = f"REM_comparison_for_patient_{hyp_file}.png"
    fig.savefig(os.path.join(save_path, out_name), dpi=150)
    plt.close(fig)
    print(f"\nFigure sauvegardée : {out_name}")