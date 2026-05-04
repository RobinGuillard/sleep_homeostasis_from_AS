import numpy as np
import pandas as pd
import torch

#WARNING le paramètre U est mal géré: on n'est pas conforme au UMP original


class Unified_for_PVT:
    """Unified sleep model computing homeostasis and sleep debt.

    This model computes the sleep homeostasis pressure (s) and sleep debt (d)
    based on sleep/wake transitions and physiological parameters. The computation
    is done using PyTorch for differentiability and GPU support.

    The model takes state changes (awake/asleep transitions) and computes the
    evolution of s and d over time using piece-wise analytic solutions for
    wake and sleep periods.

    Differential equations:
        During wake:
            ds/dt = (1 - s) / t_w
            d(d)/dt = (-d + 1) / t_la
        During sleep:
            ds/dt = -(s - d) / t_s
            d(d)/dt = (-d - wsr) / t_la
        where:
            - s: homeostasis
            - d: debt
            - t_w: time constant during wake
            - t_s: time constant during sleep
            - t_la: time constant for debt
            - wsr: (T - need) / need (wake-sleep ratio)
            - T: period
            - need: sleep need

    References
    ----------
    P. Rajdev et al., "A unified mathematical model to quantify performance impairment
    for both chronic sleep restriction and total sleep deprivation.",
    from Journal of theoretical biology. 2013.
    """

    def __init__(self):
        self.outputs = ["homeostasis", "debt", "PVT", "circadian"]  # add PVT to outputs

    def __str__(self) -> str:
        """Return a user-friendly string representation of the model."""
        return (
            "Unified sleep model for PVT estimation computing:\n"
            "- Homeostasis (s): sleep pressure\n"
            "- Debt (d): sleep debt\n\n"
            "- PVT: psychomotor vigilance test predicted\n\n"
            "During wake:\n"
            "  ds/dt = (1 - s) / t_w\n"
            "  dd/dt = (-d + 1) / t_la\n\n"
            "During sleep:\n"
            "  ds/dt = -(s - d) / t_s\n"
            "  dd/dt = (-d - wsr) / t_la\n\n"
            "Circadian time is calculated using the hour of the day t and a phase Phi:\n"
            "C= 0.97*sin(2*pi*(t+Phi)/24) + 0.22*sin(2*2*pi*(t+Phi)/24)+0.07*sin(3*2*pi*(t+Phi)/24)+0.03*sin(4*2*pi*(t+Phi)/24)+0.001*sin(5*2*pi*(t+Phi)/24)"

        )

    #not sure we are going to use this anymore
    def infer_initial_state(self, sc, rested, t_s, t_w, t_la, need, T):
        """
        Walk *backwards* through the diary until the very first timestamp,
        starting from the fully-rested (s_w, d_w) at wake - sc[rested].
        """
        wsr = (T - need) / need
        s1, d1, _, _ = self._periodic_values(t_s, t_w, t_la, wsr, T)
        gap = s1 - d1  # use the gap to find s0

        wake = False  # the first backstep segment is always sleep
        if rested > 0:
            for t0, t1 in zip(sc[:rested].flip(0), sc[1 : rested + 1].flip(0)):  # reverse iterate
                dt = (t1 - t0).reshape(1, 1)  # keep tensor shape
                if wake:
                    d1 = self._wake_backstep(d1, dt, t_la)
                else:
                    d1 = self._sleep_backstep(d1, dt, t_la, wsr)
                wake = not wake
        s1 = d1.clone() + gap

        return s1, d1  # these are s0, d0 at the very first diary row

    def compute(
        self,
        sc: torch.Tensor,
        rested: torch.Tensor,
        time: torch.Tensor,
        t_s: torch.Tensor,
        t_w: torch.Tensor,
        t_la: torch.Tensor,
        need: torch.Tensor,
        T: torch.Tensor,
        U: torch.Tensor, #scaling factor for homeostasis
        kappa: torch.Tensor, #sensitivity to circadian rhythm
        phi: torch.Tensor, #phase of the circadian rhythm 
        initial_hour: torch.Tensor = torch.Tensor([8]), #8h in the morning as default
        s0: torch.Tensor = None,
        d0: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Compute homeostasis and debt."""
        wsr = (T - need) / need
        pi, mi = wsr.numel(), time.numel() #number of elements in tensors, wsr : 1 per patient, time variable
        # ----- initialisations -----
        wake = True
        if s0 is None or d0 is None:
            s0, d0 = self.infer_initial_state(sc, rested, t_s, t_w, t_la, need, T)
        else:
            s0, d0 = s0.reshape(1, -1), d0.reshape(1, -1)  # ensure shape (1, pi)
        s = torch.full((mi, pi), torch.nan, device=time.device, dtype=time.dtype)
        d = torch.full_like(s, torch.nan)
        

        # ----- propagate over sleep/wake segments -----
        done = 0
        total = torch.sum(~torch.isnan(time)).item()
        
        for t0, t1 in zip(sc[:-1], sc[1:]):
            if done == total:
                break  # early exit if all time points computed

            in_seg = (t0 <= time) & (time < t1)
            
            last_seg = torch.isinf(t1)
            if not last_seg:  # need t1 for the next segment
                t = torch.cat([time[in_seg], t1.unsqueeze(0)])
                
            else:
                t = time[in_seg]
            
            dt = (t - t0).reshape(-1, 1)  # (mseg, 1)
            if wake:
                s_seg, d_seg = self._wake_step(s0, d0, dt, t_w, t_la, U)
            else:  # sleep
                s_seg, d_seg = self._sleep_step(s0, d0, dt, t_s, t_la, wsr, U)
            if dt.numel() > 1 or last_seg:
                s[in_seg] = s_seg[:-1] if not last_seg else s_seg
                d[in_seg] = d_seg[:-1] if not last_seg else d_seg
            done += in_seg.sum()

            s0, d0 = s_seg[-1:], d_seg[-1:]  # (1, pi)
            wake = not wake
        PVT = s + kappa * self._circadian(time, phi, initial_hour)
        circ_raw = self._circadian(time, phi, initial_hour).reshape(-1, 1)  # (mi, 1)
        circ = kappa * circ_raw  # (mi, pi)
        PVT = s + circ           # (mi, pi)
        return {"homeostasis": s.T, "debt": d.T, "PVT": PVT.T, "circadian": circ.T}  # (m, pi)

    def check_params(
        self,
        t_s: torch.Tensor,
        t_w: torch.Tensor,
        t_la: torch.Tensor,
        need: torch.Tensor,
        T: torch.Tensor,
        U: torch.Tensor,
        kappa: torch.Tensor,
        phi: torch.Tensor,
        s0: torch.Tensor = None,
        d0: torch.Tensor = None,
        initial_hour: torch.Tensor = None,
    ):
        """Validate parameters."""
        # Validate parameters
        if (t_s <= 0).any():
            raise ValueError("t_s must be positive")
        if (t_w <= 0).any():
            raise ValueError("t_w must be positive")
        if (t_la <= 0).any():
            raise ValueError("t_la must be positive")
        if (need <= 0).any():
            raise ValueError("need must be positive")
        if (T <= 0).any():
            raise ValueError("T must be positive")
        #initial_hour should be between 0 and 24 if provided, it is a tensor
        if initial_hour is not None:
            if (initial_hour < 0).any() or (initial_hour >= 24).any():
                raise ValueError("initial_hour must be in [0, 24)")
    

        # Validate relationships
        if (t_s >= t_la).any():
            raise ValueError("t_s must be < t_la")
        if (t_w >= t_la).any():
            raise ValueError("t_w must be < t_la")

    # ──────────────────────────────────────────────────────────────
    #  Core maths - all torch, differentiable
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _periodic_values(t_s, t_w, t_la, wsr, T):
        tot_w = T * wsr / (wsr + 1.0)
        tot_s = T - tot_w
        a = torch.exp(-tot_w / t_la)
        b = torch.exp(-tot_s / t_la)
        c = torch.exp(-tot_w / t_w)
        d = torch.exp(-tot_s / t_s)

        d_w = (b * (1 - a + wsr) - wsr) / (1 - a * b)
        d_s = 1 - a + a * d_w

        e = t_la / (t_la - t_s)
        s_w = (wsr * (d - 1) + d * (1 - c) + e * (b - d) * (1 - a + a * d_w + wsr)) / (1 - d * c)
        s_s = 1 - c + c * s_w
        return s_w, d_w, s_s, d_s

    @staticmethod
    def _wake_step(s0, d0, dt, t_w, t_la, U):
        """Compute the wake time point from start condition."""
        s = U - (U - s0) * torch.exp(-dt / t_w)  #  (mseg, pi)
        d = U - (U - d0) * torch.exp(-dt / t_la)
        return s, d


    #MODIFIER A PARTIR D4ICI POUR RAJOUTER U
    @staticmethod
    def _sleep_step(s0, d0, dt, t_s, t_la, wsr, U):
        """Compute the sleep time point from start condition."""
        e_s = torch.exp(-dt / t_s)
        e_la = torch.exp(-dt / t_la)
        B = t_la / (t_la - t_s)
        d = d0 * e_la - wsr* U *(1 - e_la) 
        s = s0 * e_s - wsr * U * (1 - e_s) + B * (d0 + wsr * U) * (e_la - e_s)
        return s, d

    @staticmethod
    def _wake_backstep(d1, dt, t_la):
        """Compute the wake time point from end condition."""
        d0 = 1.0 - (1.0 - d1) * torch.exp(+dt / t_la)
        return d0

    @staticmethod
    def _sleep_backstep(d1, dt, t_la, wsr):
        """Inverse of the sleep time point from end condition."""
        e_la = torch.exp(-dt / t_la)
        d0 = -wsr + (d1 + wsr) / e_la
        return d0
    @staticmethod
    def _circadian(time, phi, initial_hour):
        """Compute circadian rhythm value for given time and phase."""
        t = (time + phi + initial_hour) % 24
        C = (
            0.97 * torch.sin(2 * np.pi * t / 24)
            + 0.22 * torch.sin(2 * 2 * np.pi * t / 24)
            + 0.07 * torch.sin(3 * 2 * np.pi * t / 24)
            + 0.03 * torch.sin(4 * 2 * np.pi * t / 24)
            + 0.001 * torch.sin(5 * 2 * np.pi * t / 24)
        )
        return C

    def plot(
        self, ax, diary: pd.DataFrame, time: np.ndarray, homeostasis: np.ndarray, debt: np.ndarray, PVT: np.ndarray, circadian: np.ndarray
    ):
        """Plot homeostasis, debt, and sleep periods for a given key."""
        # Plot homeostasis and debt
        time = time / 24
        offset = time[0]  # offset to start at 0
        time -= offset
        time_hours = (time*24)  # convert to hours and add initial_hour offset
        circ = self._circadian(torch.tensor(time_hours), torch.tensor(0), torch.tensor(8))  # example with phi=0 and initial_hour=8

        ax.plot(time, homeostasis, "orange", linestyle="dotted", label="Homeostasis")
        ax.plot(time, debt, "green", label="Debt")
        ax.plot(time, PVT, "blue", label="PVT")
        ax.plot(time, circ, "purple", linestyle="dashdot", label="Circadian")
        ax.set_xlim(time[0], time[-1])
        ax.grid(True, alpha=0.3)

        for i, (asleep, awake) in enumerate(zip(diary["asleep"], diary["awake"])):
            t0 = asleep / 24 - offset
            t1 = awake / 24 - offset
            ax.axvspan(t0, t1, color="gray", alpha=0.3, label="Sleep" if i == 0 else "")


