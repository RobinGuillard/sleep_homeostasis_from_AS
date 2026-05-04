import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Constantes d'état 
# ---------------------------------------------------------------------------
WAKE = 0
NREM = 1
REM  = 4

# ---------------------------------------------------------------------------
#  REMThreeProcessCharge
# ---------------------------------------------------------------------------

class REMThreeProcessCharge:
    """Modèle d'homéostasie REM à trois états : wake, NREM, REM."""

    outputs = ["homeostasis"]

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return (
            "REMThreeProcessCharge :\n"
            "  Wake : ds/dt = (1-s)/t_w\n"
            "  NREM : ds/dt = (1-s)/t_NREM\n"
            "  REM  : ds/dt = -s/t_REM\n"
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
    #  Fractions REM par cycle (journée idéale C)
    # ------------------------------------------------------------------
    @staticmethod
    def _cycle_rem_fractions(
        REM_relative_need: torch.Tensor,   # (1, p)
        cycle_increment:   torch.Tensor,   # (1, p)
        N_cycles:          int,
    ) -> torch.Tensor:
        """Fractions de REM pour chacun des N_cycles cycles de la journée idéale.

        Formule :
          rem_frac_k = REM_relative_need * [1 + cycle_increment * (2k/(N-1) - 1)]
          pour k = 0 .. N_cycles-1

        Propriétés garanties :
          - Moyenne sur k vaut exactement REM_relative_need
            (la durée totale de REM = REM_relative_need * need)
          - Monotone croissante en k (REM concentré en fin de nuit)
          - cycle_increment = 0  ->  tous les cycles identiques
          - cycle_increment = 1  ->  frac_0 = 0, frac_{N-1} = 2*REM_relative_need

        Returns
        -------
        fracs : torch.Tensor, shape (N_cycles, p)
            Fraction de REM dans le cycle k, pour chaque jeu de paramètres.
        """
        if N_cycles == 1:
            return REM_relative_need.expand(1, -1)  # (1, p)

        k = torch.arange(N_cycles, dtype=REM_relative_need.dtype,
                         device=REM_relative_need.device)          # (N,)
        # weight_k ∈ [-1, +1], moyenne = 0 -> la moyenne de fracs = REM_relative_need
        weight = 2.0 * k / (N_cycles - 1) - 1.0                   # (N,)
        # fracs : (N, p)
        fracs = REM_relative_need * (1.0 + cycle_increment * weight.unsqueeze(1))
        return fracs.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    #  Solution périodique (journée idéale C)
    # ------------------------------------------------------------------
    @staticmethod
    def _periodic_value(
        t_w:               torch.Tensor,   # (1, p)
        t_NREM:            torch.Tensor,
        t_REM:             torch.Tensor,
        need:              torch.Tensor,
        T:                 torch.Tensor,
        REM_relative_need: torch.Tensor,
        cycle_increment:   torch.Tensor,
        N_cycles:          int,
        fracs:             torch.Tensor,   # (N_cycles, p)  pré-calculé
    ) -> torch.Tensor:
        """Point fixe s_w de la journée idéale C (solution analytique exacte).

        La journée idéale est composée de :
          - (T - need) h d'éveil               [charge, t_w]
          - N_cycles cycles de durée need/N_cycles :
              cycle k : (1-frac_k)*cycle_dur de NREM  [charge, t_NREM]
                         frac_k  *cycle_dur de REM    [décharge, t_REM]
            où frac_k = _cycle_rem_fractions()[k]

        Chaque cycle k est un opérateur affine f_k(s) = alpha_k + beta_k * s :
          alpha_k = ar_k * (1 - an_k)
          beta_k  = ar_k * an_k

        La composition séquentielle f_1 ∘ ... ∘ f_N est encore affine :
          s -> A_N + B_N * s
        avec la récurrence (initialisée à identité) :
          A_{k+1} = alpha_k + beta_k * A_k
          B_{k+1} = beta_k  * B_k

        Point fixe après wake + N cycles :
          s_w = A_N + B_N * [(1-aw) + aw*s_w]
          => s_w = (A_N + B_N*(1-aw)) / (1 - B_N*aw)
        """
        cycle_dur = need / N_cycles                          # (1, p)

        # Initialisation : opérateur identité
        A = torch.zeros_like(t_w)                            # (1, p)
        B = torch.ones_like(t_w)                             # (1, p)

        for k in range(N_cycles):
            rf    = fracs[k]                                 # (p,) -> unsqueeze -> (1, p)
            rf    = rf.unsqueeze(0) if rf.dim() == 1 else rf
            an    = torch.exp(-cycle_dur * (1.0 - rf) / t_NREM)
            ar    = torch.exp(-cycle_dur * rf          / t_REM)
            alpha = ar * (1.0 - an)
            beta  = ar * an
            A = alpha + beta * A
            B = beta  * B

        aw    = torch.exp(-(T - need) / t_w)
        num   = A + B * (1.0 - aw)
        denom = 1.0 - B * aw
        return (num / denom).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    #  Steps forward — statiques, différentiables
    # ------------------------------------------------------------------
    @staticmethod
    def _wake_step(s0: torch.Tensor, dt: torch.Tensor,
                   t_w: torch.Tensor) -> torch.Tensor:
        return 1.0 - (1.0 - s0) * torch.exp(-dt / t_w)

    @staticmethod
    def _NREM_sleep_step(s0: torch.Tensor, dt: torch.Tensor,
                         t_NREM: torch.Tensor) -> torch.Tensor:
        return 1.0 - (1.0 - s0) * torch.exp(-dt / t_NREM)

    @staticmethod
    def _REM_sleep_step(
        s0:       torch.Tensor,
        dt:       torch.Tensor,
        t_REM:    torch.Tensor,
        s_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Décharge REM vers s_target (défaut 0).

        Équation originale : ds/dt = -s / t_REM  ->  s(t) = s0 * exp(-t/t_REM)
        Généralisation     : ds/dt = -(s - s_target) / t_REM
                             ->  s(t) = s_target + (s0 - s_target) * exp(-t/t_REM)

        Avec s_target = 0 on retrouve le comportement original.
        Avec s_target = f(need, REM_relative_need, ...) le gradient traverse
        s_target et remonte vers ces paramètres à chaque segment REM.
        """
        if s_target is None:
            s_target = torch.zeros_like(s0)
        return s_target + (s0 - s_target) * torch.exp(-dt / t_REM)

    # ------------------------------------------------------------------
    #  Backsteps — statiques, différentiables
    # ------------------------------------------------------------------
    @staticmethod
    def _wake_backstep(s1: torch.Tensor, dt: torch.Tensor,
                       t_w: torch.Tensor) -> torch.Tensor:
        return 1.0 - (1.0 - s1) * torch.exp(dt / t_w)

    @staticmethod
    def _NREM_sleep_backstep(s1: torch.Tensor, dt: torch.Tensor,
                              t_NREM: torch.Tensor) -> torch.Tensor:
        return 1.0 - (1.0 - s1) * torch.exp(dt / t_NREM)

    @staticmethod
    def _REM_sleep_backstep(
        s1:       torch.Tensor,
        dt:       torch.Tensor,
        t_REM:    torch.Tensor,
        s_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inverse de _REM_sleep_step avec s_target."""
        if s_target is None:
            s_target = torch.zeros_like(s1)
        return s_target + (s1 - s_target) * torch.exp(dt / t_REM)

    # ------------------------------------------------------------------
    #  Inférence de l'état initial (backstep depuis rested_idx -> 0)
    # ------------------------------------------------------------------
    def infer_initial_state(
        self,
        sc_times:          torch.Tensor,   # (n_transitions,)
        sc_states:         torch.Tensor,   # (n_transitions,)  {0, 1, 4}
        rested_idx:        int,
        t_w:               torch.Tensor,   # (1, p)
        t_NREM:            torch.Tensor,
        t_REM:             torch.Tensor,
        need:              torch.Tensor,
        T:                 torch.Tensor,
        REM_relative_need: torch.Tensor,
        cycle_increment:   torch.Tensor,
        N_cycles:          int,
    ) -> torch.Tensor:
        """Remonte depuis sc_times[rested_idx] jusqu'a sc_times[0].

        Part de la valeur périodique s_w (début d'un éveil reposé) et
        applique les backsteps correspondant à l'état de chaque segment
        parcouru en sens inverse.

        Returns
        -------
        s0 : torch.Tensor, shape (1, p)
            Homéostasie au temps sc_times[0].
        """
        fracs = self._cycle_rem_fractions(REM_relative_need, cycle_increment, N_cycles)
        s1 = self._periodic_value(
            t_w, t_NREM, t_REM, need, T,
            REM_relative_need, cycle_increment, N_cycles, fracs,
        )

        # s_target REM : valeur d'équilibre vers laquelle décharge le REM.
        # Ici on utilise 0 comme dans l'équation originale, mais on le passe
        # explicitement pour que le graphe de calcul reste connecté si on
        # souhaite à terme le paramétrer (ex. s_target = f(need, ...)).
        s_target = torch.zeros_like(s1)  # (1, p)  -- différentiable, grad=0 pour l'instant
        #s_target = (REM_relative_need * need - T) / (REM_relative_need * need)

        if rested_idx > 0:
            # Segments à remonter : [0 .. rested_idx], parcourus à l'envers
            times_rev  = sc_times[:rested_idx + 1].flip(0)
            states_rev = sc_states[:rested_idx + 1].flip(0)

            for k in range(len(times_rev) - 1):
                t_end   = times_rev[k]
                t_start = times_rev[k + 1]
                dt      = (t_end - t_start).reshape(1, 1)
                state   = states_rev[k].item()

                # Stabilisation numérique sans couper le gradient :
                # on utilise sigmoid pour rester dans (0,1) de façon différentiable.
                s1 = torch.sigmoid(torch.logit(s1.clamp(1e-6, 1 - 1e-6)))

                if state == WAKE:
                    s1 = self._wake_backstep(s1, dt, t_w)
                elif state == NREM:
                    s1 = self._NREM_sleep_backstep(s1, dt, t_NREM)
                elif state == REM:
                    s1 = self._REM_sleep_backstep(s1, dt, t_REM, s_target)

        # Normalisation finale différentiable via sigmoid(logit(.)).
        # Contrairement à clamp(0,1), sigmoid ne tue pas le gradient quand s1
        # sort de [0,1] (ce qui arrive fréquemment avec _REM_sleep_backstep car
        # l'exponentielle croissante exp(+dt/t_REM) peut faire exploser s1).
        # Cela permet à need, T, REM_relative_need et cycle_increment de
        # conserver un gradient non nul via _periodic_value -> s1 -> s0.
        return torch.sigmoid(s1)  # (1, p)  -- équivalent à clamp smooth

    # ------------------------------------------------------------------
    #  Propagation forward
    # ------------------------------------------------------------------
    def compute(
        self,
        sc_times:          torch.Tensor,   # (n_transitions,)
        sc_states:         torch.Tensor,   # (n_transitions,)  {0, 1, 4}
        s0:                torch.Tensor,   # (1, p)  condition initiale
        time:              torch.Tensor,   # (mi,)
        t_w:               torch.Tensor,   # (1, p)
        t_NREM:            torch.Tensor,
        t_REM:             torch.Tensor,
        need:              torch.Tensor,
        T:                 torch.Tensor,
        REM_relative_need: torch.Tensor,
        cycle_increment:   torch.Tensor,
        N_cycles:          int,
    ) -> dict[str, torch.Tensor]:
        """Calcule l'homéostasie à chaque instant de `time`.

        s0 est la condition initiale en sc_times[0], fournie par l'appelant
        (soit via infer_initial_state en mode "stable", soit directement en
        mode "free"). Cela permet à s0 de porter un requires_grad=True
        indépendant du backstep.

        Pour chaque segment [sc_times[k], sc_times[k+1]), l'état sc_states[k]
        détermine le step appliqué :
          WAKE (0) -> _wake_step
          NREM (1) -> _NREM_sleep_step
          REM  (4) -> _REM_sleep_step

        Returns
        -------
        {"homeostasis": torch.Tensor, shape (p, mi)}
        """
        mi = time.numel()
        p  = 1  # nombre de jeux de paramètres (actuellement 1, mais peut être >1 pour batch ou grid search)

        # s_target REM : idem infer_initial_state, zéro par défaut.
        # Si on souhaite le paramétrer plus tard (ex. valeur résiduelle non nulle),
        # il suffit de le calculer depuis need/REM_relative_need ici.
        s_target = torch.zeros(1, p, device=time.device, dtype=time.dtype)
        #s_target = (REM_relative_need * need - T) / (REM_relative_need * need)

        # Détermine une fois si le graphe de calcul doit être maintenu
        any_grad = any(
            v.requires_grad for v in [s0, t_w, t_NREM, t_REM, need, T,
                                      REM_relative_need, cycle_increment]
        )

        s = torch.full((mi, p), float("nan"), device=time.device, dtype=time.dtype)

        valid_mask  = ~torch.isnan(time)
        total_valid = valid_mask.sum().item()
        done        = 0

        # Sentinelle +inf pour clore le dernier segment
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

            dt = (t_eval - t_seg_start).reshape(-1, 1)  # (mseg[+1], 1)

            if state == WAKE:
                s_seg = self._wake_step(s0, dt, t_w)
            elif state == NREM:
                s_seg = self._NREM_sleep_step(s0, dt, t_NREM)
            elif state == REM:
                s_seg = self._REM_sleep_step(s0, dt, t_REM, s_target)
            else:
                raise ValueError(f"État inconnu : {state}")

            if in_seg.any():
                n_in = in_seg.sum().item()
                s[in_seg] = s_seg[:n_in]
                done += n_in

            # Mise à jour de s0 : dernier point calculé (y compris t_seg_end)
            if dt.numel() > 0:
                s0 = s_seg[-1:] if any_grad else s_seg[-1:].detach()

        return {"homeostasis": s.T}  # (p, mi)

    # ------------------------------------------------------------------
    #  Visualisation (appelée par ModelHypnogram.plot)
    # ------------------------------------------------------------------
    def plot(
        self,
        ax,
        sc_times:    np.ndarray,
        sc_states:   np.ndarray,
        time:        np.ndarray,
        homeostasis: np.ndarray,
        rested_idx:  int | None = None,   # None -> pas de ligne verticale
    ):
        time_days = time / 24
        offset    = time_days[0]
        time_days = time_days - offset

        ax.plot(time_days, homeostasis, color="green", label="Homéostasie")

        # Ligne verticale au point reposé (uniquement si rested_idx est fourni)
        if rested_idx is not None:
            rest_time = sc_times[rested_idx] / 24 - offset
            ax.axvline(rest_time, color="red", linestyle="--", label="Point reposé")

        ax.set_xlim(time_days[0], time_days[-1])
        ax.grid(True, alpha=0.3)

        # Coloration des segments sommeil
        colors = {NREM: ("gray", "NREM"), REM: ("gold", "REM")}
        shown  = set()
        for k in range(len(sc_times)):
            t0    = sc_times[k] / 24 - offset
            t1    = (sc_times[k + 1] / 24 - offset) if k + 1 < len(sc_times) else time_days[-1]
            state = int(sc_states[k])
            if state in colors:
                color, lbl = colors[state]
                label = lbl if state not in shown else ""
                ax.axvspan(t0, t1, color=color, alpha=0.25, label=label)
                shown.add(state)

