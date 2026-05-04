import numpy as np
import torch

# ---------------------------------------------------------------------------
#  Constantes d'état (identiques à REMThreeProcessCharge)
# ---------------------------------------------------------------------------
WAKE = 0
NREM = 1
REM  = 4


# ---------------------------------------------------------------------------
#  REMThreeProcessDischarge
# ---------------------------------------------------------------------------
class REMThreeProcessNeutral:
    """Modèle d'homéostasie REM à trois états : wake, NREM, REM.

    Variante "discharge" : la pression REM se DÉCHARGE durant l'éveil
    (et durant le REM), et se CHARGE durant le NREM.

    Équations différentielles
    -------------------------
      Wake : ds/dt = -(s - s_target_wake) / t_w      [décharge vers s_target_wake]
      NREM : ds/dt =  (1 - s)             / t_NREM   [charge vers 1]
      REM  : ds/dt = -(s - s_target_REM)  / t_REM    [décharge vers s_target_REM]

    Par défaut :
      s_target_wake = 0   (décharge complète vers 0 en éveil)
      s_target_REM  = 0   (décharge complète vers 0 en REM)

    Ces deux cibles peuvent être paramétrées pour rendre need, T et
    REM_relative_need différentiables (même logique que REMThreeProcessCharge).

    Solution périodique (journée idéale)
    -------------------------------------
    La journée idéale est :
      - (T - need) h d'éveil       [décharge, t_w]
      - N_cycles cycles de need/N_cycles :
          (1 - frac_k) * cycle_dur de NREM  [charge, t_NREM]
          frac_k       * cycle_dur de REM   [décharge, t_REM]

    Chaque opérateur est affine : f(s) = alpha + beta * s
      Wake  :  alpha_w  = s_target_wake * (1 - aw),   beta_w  = aw
               avec aw = exp(-(T-need)/t_w)
      NREM  :  alpha_n  = (1 - an),                   beta_n  = an
               avec an = exp(-dt_NREM/t_NREM)
      REM   :  alpha_r  = s_target_REM * (1 - ar),    beta_r  = ar
               avec ar = exp(-dt_REM/t_REM)

    On compose les N_cycles cycles puis l'éveil, on cherche le point fixe.
    """

    outputs = ["homeostasis"]

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return (
            "REMThreeProcessNeutral :\n"
            "  Wake : ds/dt = 0                               [constant]\n"
            "  NREM : ds/dt =  (1 - s)             / t_NREM  [charge]\n"
            "  REM  : ds/dt = -(s - s_target_REM)  / t_REM   [décharge]\n"
        )

    # ------------------------------------------------------------------
    def check_params(
        self,
        t_w:               torch.Tensor,
        t_NREM:            torch.Tensor,
        t_REM:             torch.Tensor,
        need:              torch.Tensor,
        T:                 torch.Tensor,
        REM_relative_need: torch.Tensor,
        cycle_increment:   torch.Tensor,
    ):
        for name, val in [
            ("t_w",               t_w),
            ("t_NREM",            t_NREM),
            ("t_REM",             t_REM),
            ("need",              need),
            ("T",                 T),
            ("REM_relative_need", REM_relative_need),
        ]:
            if (val <= 0).any():
                raise ValueError(f"{name} doit être strictement positif")
        if (REM_relative_need >= 1).any():
            raise ValueError("REM_relative_need doit être < 1")
        if (cycle_increment < 0).any() or (cycle_increment > 1).any():
            raise ValueError("cycle_increment doit être dans [0, 1]")

    # ------------------------------------------------------------------
    #  Fractions REM par cycle  (identique à REMThreeProcessCharge)
    # ------------------------------------------------------------------
    @staticmethod
    def _cycle_rem_fractions(
        REM_relative_need: torch.Tensor,   # (1, p)
        cycle_increment:   torch.Tensor,   # (1, p)
        N_cycles:          int,
    ) -> torch.Tensor:
        """Fractions de REM pour chacun des N_cycles cycles (moy = REM_relative_need)."""
        if N_cycles == 1:
            return REM_relative_need.expand(1, -1)
        k      = torch.arange(N_cycles, dtype=REM_relative_need.dtype,
                               device=REM_relative_need.device)
        weight = 2.0 * k / (N_cycles - 1) - 1.0                    # (N,) ∈ [-1,+1]
        fracs  = REM_relative_need * (1.0 + cycle_increment * weight.unsqueeze(1))
        return fracs.clamp(0.0, 1.0)                                # (N, p)

    # ------------------------------------------------------------------
    #  Solution périodique
    # ------------------------------------------------------------------
    @staticmethod
    def _periodic_value(
        t_NREM:            torch.Tensor,
        t_REM:             torch.Tensor,
        need:              torch.Tensor,
        REM_relative_need: torch.Tensor,
        cycle_increment:   torch.Tensor,
        N_cycles:          int,
        fracs:             torch.Tensor,   # (N_cycles, p)
        s_target_REM=torch.Tensor([0]),   # (1, p)
    ) -> torch.Tensor:
        """Point fixe de la journée idéale (éveil = identité).

        Durant l'éveil ds/dt=0, donc l'opérateur wake est l'identité.
        T et t_w n'interviennent pas dans le point fixe.
        Point fixe : s = A_cycles / (1 - B_cycles)
        """
        cycle_dur = need / N_cycles   # (1, p)

        # Composition des N cycles (NREM puis REM pour chaque cycle k)
        # L'opérateur wake est l'identité (ds/dt=0), donc pas de terme aw ici.
        A = torch.zeros_like(t_NREM)  # (1, p)
        B = torch.ones_like(t_NREM)   # (1, p)

        for k in range(N_cycles):
            rf = fracs[k].unsqueeze(0) if fracs[k].dim() == 1 else fracs[k]
            dt_NREM_k = cycle_dur * (1.0 - rf)
            dt_REM_k  = cycle_dur * rf

            an      = torch.exp(-dt_NREM_k / t_NREM)
            ar      = torch.exp(-dt_REM_k  / t_REM)
            alpha_n = 1.0 - an
            alpha_r = s_target_REM * (1.0 - ar)

            alpha_cycle = alpha_r + ar * alpha_n
            beta_cycle  = ar * an

            A = alpha_cycle + beta_cycle * A
            B = beta_cycle  * B

        # Point fixe : wake = identité => s = A + B*s => s = A / (1 - B)
        return A / (1.0 - B)   # (1, p)

    # ------------------------------------------------------------------
    #  Steps forward
    # ------------------------------------------------------------------
    @staticmethod
    def _wake_neutral_step(
        s0: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Durant l'éveil, s reste constant : ds/dt = 0 => s(t) = s0."""
        return s0.expand(dt.shape[0], -1)   # (mseg, p)

    @staticmethod
    def _NREM_sleep_step(
        s0:    torch.Tensor,
        dt:    torch.Tensor,
        t_NREM: torch.Tensor,
    ) -> torch.Tensor:
        """Charge en NREM vers 1 (identique à REMThreeProcessCharge)."""
        return 1.0 - (1.0 - s0) * torch.exp(-dt / t_NREM)

    @staticmethod
    def _REM_sleep_step(
        s0:           torch.Tensor,
        dt:           torch.Tensor,
        t_REM:        torch.Tensor,
        s_target_REM: torch.Tensor,
    ) -> torch.Tensor:
        """Décharge en REM vers s_target_REM.

        ds/dt = -(s - s_target_REM) / t_REM
        s(t)  = s_target_REM + (s0 - s_target_REM) * exp(-dt/t_REM)
        """
        return s_target_REM + (s0 - s_target_REM) * torch.exp(-dt / t_REM)

    # ------------------------------------------------------------------
    #  Backsteps
    # ------------------------------------------------------------------
    @staticmethod
    def _wake_neutral_backstep(
        s1: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse du wake neutral : s constant, donc s0 = s1."""
        return s1

    @staticmethod
    def _NREM_sleep_backstep(
        s1:    torch.Tensor,
        dt:    torch.Tensor,
        t_NREM: torch.Tensor,
    ) -> torch.Tensor:
        return 1.0 - (1.0 - s1) * torch.exp(dt / t_NREM)

    @staticmethod
    def _REM_sleep_backstep(
        s1:           torch.Tensor,
        dt:           torch.Tensor,
        t_REM:        torch.Tensor,
        s_target_REM: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse du REM discharge step."""
        return s_target_REM + (s1 - s_target_REM) * torch.exp(dt / t_REM)

    # ------------------------------------------------------------------
    #  Inférence de la condition initiale (backstep depuis rested_idx)
    # ------------------------------------------------------------------
    def infer_initial_state(
        self,
        sc_times:          torch.Tensor,
        sc_states:         torch.Tensor,
        rested_idx:        int,
        t_NREM:            torch.Tensor,
        t_REM:             torch.Tensor,
        need:              torch.Tensor,
        REM_relative_need: torch.Tensor,
        cycle_increment:   torch.Tensor,
        N_cycles:          int,
        s_target_REM=torch.Tensor([0]),
        t_w: torch.Tensor | None = None,
        T:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        fracs = self._cycle_rem_fractions(REM_relative_need, cycle_increment, N_cycles)
        s1    = self._periodic_value(
            t_NREM, t_REM, need,
            REM_relative_need, cycle_increment, N_cycles, fracs,
            s_target_REM,
        )

        if rested_idx > 0:
            times_rev  = sc_times[:rested_idx + 1].flip(0)
            states_rev = sc_states[:rested_idx + 1].flip(0)

            for k in range(len(times_rev) - 1):
                t_end_k   = times_rev[k]
                t_start_k = times_rev[k + 1]
                dt        = (t_end_k - t_start_k).reshape(1, 1)
                state     = states_rev[k].item()

                if state == WAKE:
                    s1 = self._wake_neutral_backstep(s1, dt)   # identité
                elif state == NREM:
                    s1 = self._NREM_sleep_backstep(s1, dt, t_NREM)
                elif state == REM:
                    s1 = self._REM_sleep_backstep(s1, dt, t_REM, s_target_REM)

        return s1  # (1, p)

    # ------------------------------------------------------------------
    #  Propagation forward
    # ------------------------------------------------------------------
    def compute(
        self,
        sc_times:          torch.Tensor,
        sc_states:         torch.Tensor,
        s0:                torch.Tensor,   # (1, p)
        time:              torch.Tensor,   # (mi,)
        t_NREM:            torch.Tensor,
        t_REM:             torch.Tensor,
        need:              torch.Tensor,
        REM_relative_need: torch.Tensor,
        cycle_increment:   torch.Tensor,
        N_cycles:          int,
        s_target_REM:      torch.Tensor | None = None,
        # t_w et T acceptés mais ignorés (pour compatibilité d'interface)
        t_w:               torch.Tensor | None = None,
        T:                 torch.Tensor  | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calcule l'homéostasie. t_w et T sont ignorés (ds/dt=0 en wake).

        Returns
        -------
        {"homeostasis": torch.Tensor, shape (p, mi)}
        """
        mi = time.numel()
        p  = t_NREM.shape[1] if t_NREM.dim() > 1 else 1

        if s_target_REM is None:
            s_target_REM = torch.zeros(1, p, device=time.device, dtype=time.dtype)

        any_grad = any(
            v.requires_grad for v in [s0, t_NREM, t_REM, need,
                                      REM_relative_need, cycle_increment]
        )

        s           = torch.full((mi, p), float("nan"), device=time.device, dtype=time.dtype)
        valid_mask  = ~torch.isnan(time)
        total_valid = valid_mask.sum().item()
        done        = 0

        inf_sentinel = torch.tensor([float("inf")], device=sc_times.device,
                                    dtype=sc_times.dtype)
        times_ext  = torch.cat([sc_times, inf_sentinel])
        states_ext = torch.cat([sc_states, sc_states[-1:]])

        for k in range(len(sc_times)):
            if done >= total_valid:
                break

            t_seg_start = times_ext[k]
            t_seg_end   = times_ext[k + 1]
            state       = states_ext[k].item()
            last_seg    = torch.isinf(t_seg_end)

            in_seg = (time >= t_seg_start) & (time < t_seg_end) & valid_mask

            if not last_seg:
                t_eval = torch.cat([time[in_seg], t_seg_end.unsqueeze(0)])
            else:
                t_eval = time[in_seg]

            dt = (t_eval - t_seg_start).reshape(-1, 1)

            if state == WAKE:
                s_seg = self._wake_neutral_step(s0, dt)        # s constant
            elif state == NREM:
                s_seg = self._NREM_sleep_step(s0, dt, t_NREM)
            elif state == REM:
                s_seg = self._REM_sleep_step(s0, dt, t_REM, s_target_REM)
            else:
                raise ValueError(f"État inconnu : {state}")

            if in_seg.any():
                n_in = in_seg.sum().item()
                s[in_seg] = s_seg[:n_in]
                done += n_in

            if dt.numel() > 0:
                s0 = s_seg[-1:] if any_grad else s_seg[-1:].detach()

        return {"homeostasis": s.T}   # (p, mi)

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------
    def plot(
        self,
        ax,
        sc_times:    np.ndarray,
        sc_states:   np.ndarray,
        time:        np.ndarray,
        homeostasis: np.ndarray,
        rested_idx:  int | None = None,
    ):
        time_days = time / 24 - time[0] / 24

        ax.plot(time_days, homeostasis, color="mediumpurple", label="Homéostasie (neutral)")

        if rested_idx is not None:
            rest_time = sc_times[rested_idx] / 24 - time[0] / 24
            ax.axvline(rest_time, color="red", linestyle="--", label="Point reposé")

        ax.set_xlim(time_days[0], time_days[-1])
        ax.grid(True, alpha=0.3)

        colors = {NREM: ("gray", "NREM"), REM: ("gold", "REM")}
        shown  = set()
        for k in range(len(sc_times)):
            t0    = sc_times[k] / 24 - time[0] / 24
            t1    = (sc_times[k + 1] / 24 - time[0] / 24) if k + 1 < len(sc_times) else time_days[-1]
            state = int(sc_states[k])
            if state in colors:
                color, lbl = colors[state]
                label = lbl if state not in shown else ""
                ax.axvspan(t0, t1, color=color, alpha=0.25, label=label)
                shown.add(state)