"""
=============================================================================
  MODULE 3 — Predictive Handoff with Blockage Detection
=============================================================================
Features:
  • Mobility-trajectory prediction (Kalman filter)
  • SNR time-series forecasting (Holt's double-exponential smoothing)
  • Dynamic blockage detection (rapid SNR-drop detector)
  • Proactive handoff: triggers HO before blockage causes outage
  • ReactiveHandoffEngine — the non-predictive baseline (Reactive-HO baseline) that
    Module 3 is benchmarked against in Section IV
=============================================================================

  >>> Tuning-history note <<<
  The constants below (HO_TRIGGER_MS, COOLDOWN_TICKS, the confidence
  formula) were originally scattered through the code as inline "FIX:"
  comments documenting a manual tuning history (e.g. "only fire HO when
  outage is within 3 ticks, not 10"). They are consolidated here as
  named class constants with the *reasoning* kept as a docstring/comment
  next to each one, so the current values are traceable design choices
  rather than leftover edit trails.
=============================================================================
"""

import math
import time
import random
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Dict, Optional
from enum import Enum

from multiband_planning import Band, BaseStation, UserEquipment, Position, PropagationModel


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Kalman filter for UE position & velocity tracking
# ─────────────────────────────────────────────────────────────────────────────

class KalmanTracker:
    """
    2-D constant-velocity Kalman filter.

    State vector: [x, y, vx, vy]
    Observation:  [x, y]  (GPS / positioning)
    """
    def __init__(self, pos: Position, dt: float = 0.1):
        self.dt = dt
        self.x = np.array([pos.x, pos.y, 0.0, 0.0], dtype=float)

        self.F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        q = 0.5   # process noise intensity
        self.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2,   0],
            [0, dt**3/2, 0, dt**2   ],
        ], dtype=float)

        self.R = np.diag([4.0, 4.0])   # measurement noise (GPS accuracy ~2 m)
        self.P = np.eye(4) * 100.0     # initial error covariance

    def predict(self) -> np.ndarray:
        """Propagate state forward one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Incorporate new GPS measurement [x, y]."""
        z = measurement
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x

    def predict_positions(self, steps: int) -> List[np.ndarray]:
        """Return future predicted positions for `steps` time steps, WITHOUT mutating self.x/P."""
        x_tmp = self.x.copy()
        positions = []
        for _ in range(steps):
            x_tmp = self.F @ x_tmp
            positions.append(x_tmp[:2].copy())
        return positions


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  SNR time-series predictor (double-exponential smoothing)
# ─────────────────────────────────────────────────────────────────────────────

class SNRPredictor:
    """
    Holt's double-exponential smoothing to forecast SNR trajectory.
    Also detects sudden blockage events (rapid SNR drops).
    """

    WINDOW_SIZE    = 20    # samples kept in ring buffer
    DROP_THRESHOLD = 8.0   # dB drop per sample indicating blockage
    TREND_HORIZON  = 5     # forecast steps ahead

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha    # level smoothing
        self.beta  = beta     # trend smoothing
        self.level = {}       # band -> current level
        self.trend = {}       # band -> current trend
        self.history = {b: deque(maxlen=self.WINDOW_SIZE) for b in Band}
        self.blockage_flags: Dict[Band, bool] = {b: False for b in Band}

    def update(self, band: Band, snr_db: float):
        """Ingest a new SNR sample."""
        self.history[band].append(snr_db)
        h = self.history[band]
        if len(h) < 2:
            self.level[band] = snr_db
            self.trend[band] = 0.0
            return

        prev_level = self.level.get(band, h[-2])
        prev_trend = self.trend.get(band, 0.0)

        new_level = self.alpha * snr_db + (1 - self.alpha) * (prev_level + prev_trend)
        new_trend = self.beta * (new_level - prev_level) + (1 - self.beta) * prev_trend

        self.level[band] = new_level
        self.trend[band] = new_trend

        drop = h[-2] - snr_db
        self.blockage_flags[band] = drop > self.DROP_THRESHOLD

    def forecast(self, band: Band, steps: int = 5) -> List[float]:
        """Return `steps` future SNR forecasts."""
        if band not in self.level:
            return []
        lvl = self.level[band]
        trd = self.trend[band]
        return [lvl + trd * t for t in range(1, steps + 1)]

    def is_blockage_imminent(self, band: Band, snr_threshold: float = 5.0) -> bool:
        """True if predicted SNR will drop below threshold within the forecast horizon."""
        forecast = self.forecast(band, self.TREND_HORIZON)
        if not forecast:
            return False
        return any(f < snr_threshold for f in forecast) or self.blockage_flags[band]


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Blockage geometry detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Obstacle:
    """Rectangular obstacle in the 2D plane."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    attenuation_db: float = 25.0   # additional signal loss through obstacle


class BlockageGeometryDetector:
    """
    Determines if the line-of-sight between a BS and a predicted UE
    position is blocked by any registered obstacle.
    """
    def __init__(self):
        self.obstacles: List[Obstacle] = []

    def add_obstacle(self, obs: Obstacle):
        self.obstacles.append(obs)

    def _segment_intersects_rect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        obs: Obstacle,
    ) -> bool:
        """Liang-Barsky line-rectangle intersection test."""
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1

        def clip(num, den, t_min, t_max):
            if abs(den) < 1e-10:
                return (num >= 0, t_min, t_max)
            t = num / den
            if den < 0:
                t_max = min(t_max, t)
            else:
                t_min = max(t_min, t)
            return t_min <= t_max, t_min, t_max

        t_min, t_max = 0.0, 1.0
        ok, t_min, t_max = clip(obs.x_min - x1,  dx, t_min, t_max)
        if not ok:
            return False
        ok, t_min, t_max = clip(x1 - obs.x_max, -dx, t_min, t_max)
        if not ok:
            return False
        ok, t_min, t_max = clip(obs.y_min - y1,  dy, t_min, t_max)
        if not ok:
            return False
        ok, t_min, t_max = clip(y1 - obs.y_max, -dy, t_min, t_max)
        if not ok:
            return False
        return True

    def blockage_loss_db(
        self,
        bs: BaseStation,
        ue_pos: Tuple[float, float],
    ) -> float:
        """Total extra attenuation (dB) due to obstacles in the LoS path."""
        p1 = (bs.position.x, bs.position.y)
        total_loss = 0.0
        for obs in self.obstacles:
            if self._segment_intersects_rect(p1, ue_pos, obs):
                total_loss += obs.attenuation_db
        return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  Predictive handoff engine (proposed Module 3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictiveHandoffDecision:
    ue_id:               int
    trigger_time_ms:     float
    source_band:         Band
    target_band:         Band
    reason:              str
    confidence:          float          # [0, 1]
    predicted_snr_at_ho: float          # dB
    time_to_outage_ms:   float          # estimated ms before link failure


class PredictiveHandoffEngine:
    """
    Orchestrates Kalman tracking + SNR prediction + geometry blockage
    detection to issue proactive handoff commands BEFORE link failure
    occurs.
    """

    FORECAST_STEPS = 10      # ticks of look-ahead (10 x TICK_MS = 1000 ms horizon)
    TICK_MS        = 100.0
    SNR_OUTAGE_DB  = 3.0
    HO_PREP_MS     = 20.0

    # Only fire a proactive HO when the forecast outage falls within this
    # window; firing on every horizon hit (up to 1000 ms out) produced far
    # too many speculative handoffs against forecasts that hadn't
    # materialized by the time they'd have executed.
    HO_TRIGGER_MS  = 300.0

    # Minimum ticks between two handoffs for the same UE — prevents
    # ping-ponging when a UE sits near a blockage/coverage boundary and
    # the forecast oscillates around the trigger threshold tick to tick.
    COOLDOWN_TICKS = 5

    def __init__(self, bss: List[BaseStation]):
        self.bss             = bss
        self.prop            = PropagationModel()
        self.trackers:       Dict[int, KalmanTracker] = {}
        self.snr_predictors: Dict[int, SNRPredictor]  = {}
        self.geo_detector    = BlockageGeometryDetector()
        self.handoff_log:    List[PredictiveHandoffDecision] = []
        self._cooldown_remaining: Dict[int, int] = {}

    def register_ue(self, ue: UserEquipment):
        self.trackers[ue.ue_id]       = KalmanTracker(ue.position, dt=self.TICK_MS / 1000.0)
        self.snr_predictors[ue.ue_id] = SNRPredictor()

    def add_obstacle(self, obs: Obstacle):
        self.geo_detector.add_obstacle(obs)

    def tick(self, ue: UserEquipment, time_ms: float) -> Optional[PredictiveHandoffDecision]:
        """
        Process one simulation tick for a UE.
        Returns a PredictiveHandoffDecision if a proactive HO is required.
        """
        tracker   = self.trackers.get(ue.ue_id)
        predictor = self.snr_predictors.get(ue.ue_id)
        if tracker is None or predictor is None:
            return None

        cd = self._cooldown_remaining.get(ue.ue_id, 0)
        if cd > 0:
            self._cooldown_remaining[ue.ue_id] = cd - 1
            return None

        # --- Update Kalman with noisy GPS measurement ---
        noisy_pos = np.array([
            ue.position.x + random.gauss(0, 1.0),
            ue.position.y + random.gauss(0, 1.0),
        ])
        tracker.predict()
        tracker.update(noisy_pos)

        # --- Update SNR history for the serving band ---
        if ue.assigned_band and ue.assigned_bs is not None:
            serving_bs = next((b for b in self.bss if b.bs_id == ue.assigned_bs), None)
            if serving_bs:
                snr = self.prop.snr_db(serving_bs, ue)
                predictor.update(ue.assigned_band, snr)

        if ue.assigned_band is None:
            return None

        # --- Check if blockage is imminent on serving band ---
        imminent = predictor.is_blockage_imminent(ue.assigned_band)
        if not imminent:
            future_positions = tracker.predict_positions(self.FORECAST_STEPS)
            serving_bs = next((b for b in self.bss if b.bs_id == ue.assigned_bs), None)
            if serving_bs:
                for fp in future_positions:
                    loss = self.geo_detector.blockage_loss_db(serving_bs, (fp[0], fp[1]))
                    if loss > 15:
                        imminent = True
                        break

        if not imminent:
            return None

        # --- Find best alternative band ---
        best_band, best_snr = None, -99.0
        for bs in self.bss:
            if bs.band == ue.assigned_band:
                continue
            snr = self.prop.snr_db(bs, ue)
            if snr > best_snr:
                best_snr, best_band = snr, bs.band

        if best_band is None:
            return None

        # --- Estimate time to outage on current band ---
        forecast = predictor.forecast(ue.assigned_band, self.FORECAST_STEPS)
        time_to_outage = self.FORECAST_STEPS * self.TICK_MS  # default: full horizon
        for step, f_snr in enumerate(forecast):
            if f_snr < self.SNR_OUTAGE_DB:
                time_to_outage = (step + 1) * self.TICK_MS
                break

        if time_to_outage < self.HO_PREP_MS:
            reason = "REACTIVE - outage already imminent"
        elif time_to_outage < 3 * self.TICK_MS:
            reason = "PROACTIVE - blockage detected, pre-emptive HO"
        else:
            reason = "PREDICTIVE - trajectory-based blockage forecast"

        if time_to_outage > self.HO_TRIGGER_MS:
            return None

        # Confidence: 1.0 = outage right now, 0.0 = outage exactly at the
        # trigger-window boundary.
        confidence = 1.0 - (time_to_outage / self.HO_TRIGGER_MS)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        decision = PredictiveHandoffDecision(
            ue_id=ue.ue_id, trigger_time_ms=time_ms,
            source_band=ue.assigned_band, target_band=best_band,
            reason=reason, confidence=confidence,
            predicted_snr_at_ho=best_snr, time_to_outage_ms=time_to_outage,
        )
        self.handoff_log.append(decision)
        self._cooldown_remaining[ue.ue_id] = self.COOLDOWN_TICKS
        return decision

    def print_decision(self, d: PredictiveHandoffDecision):
        print(f"\n  PREDICTIVE HO | UE {d.ue_id} | t={d.trigger_time_ms:.0f} ms")
        print(f"    {d.source_band.value} -> {d.target_band.value}")
        print(f"    Reason     : {d.reason}")
        print(f"    Confidence : {d.confidence*100:.1f}%")
        print(f"    Est. SNR at target : {d.predicted_snr_at_ho:.1f} dB")
        print(f"    Time to outage     : {d.time_to_outage_ms:.0f} ms")

    def summary(self):
        print("\n" + "=" * 60)
        print("  PREDICTIVE HANDOFF ENGINE — SUMMARY")
        print("=" * 60)
        print(f"  Total HO decisions : {len(self.handoff_log)}")
        if self.handoff_log:
            confs = [d.confidence for d in self.handoff_log]
            print(f"  Avg confidence     : {np.mean(confs)*100:.1f}%")
            reasons = {}
            for d in self.handoff_log:
                key = d.reason.split("-")[0].strip()
                reasons[key] = reasons.get(key, 0) + 1
            print("  HO type breakdown  :")
            for k, v in reasons.items():
                print(f"    {k}: {v}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  Reactive-handoff baseline (not a numbered paper baseline) — reactive-only handoff engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReactiveHandoffDecision:
    ue_id:           int
    trigger_time_ms: float
    source_band:     Band
    target_band:     Band
    snr_at_trigger:  float   # dB, serving-band SNR at the moment HO fired


class ReactiveHandoffEngine:
    """
    Baseline (Reactive-HO baseline): non-predictive, threshold-crossing handoff.

    No Kalman trajectory prediction, no SNR forecasting, no blockage
    geometry lookahead — this engine only reacts AFTER the serving
    band's *measured* SNR has already dropped below `SNR_OUTAGE_DB`,
    which is exactly the failure mode PredictiveHandoffEngine is
    designed to avoid. Used to reproduce the paper's reported handover
    rate / ping-pong rate comparison against Reactive-HO baseline (Section IV).

    Shares SNR_OUTAGE_DB and COOLDOWN_TICKS with PredictiveHandoffEngine
    so the only difference between the two engines is proactive vs.
    reactive triggering, not a difference in outage/cooldown definitions.
    """

    SNR_OUTAGE_DB  = PredictiveHandoffEngine.SNR_OUTAGE_DB
    TICK_MS        = PredictiveHandoffEngine.TICK_MS
    COOLDOWN_TICKS = PredictiveHandoffEngine.COOLDOWN_TICKS

    def __init__(self, bss: List[BaseStation]):
        self.bss = bss
        self.prop = PropagationModel()
        self.handoff_log: List[ReactiveHandoffDecision] = []
        self._cooldown_remaining: Dict[int, int] = {}

    def register_ue(self, ue: UserEquipment):
        # No per-UE state needed beyond the cooldown counter.
        self._cooldown_remaining.setdefault(ue.ue_id, 0)

    def tick(self, ue: UserEquipment, time_ms: float) -> Optional[ReactiveHandoffDecision]:
        if ue.assigned_band is None or ue.assigned_bs is None:
            return None

        cd = self._cooldown_remaining.get(ue.ue_id, 0)
        if cd > 0:
            self._cooldown_remaining[ue.ue_id] = cd - 1
            return None

        serving_bs = next((b for b in self.bss if b.bs_id == ue.assigned_bs), None)
        if serving_bs is None:
            return None

        snr = self.prop.snr_db(serving_bs, ue)
        if snr >= self.SNR_OUTAGE_DB:
            return None   # link is still fine — react to nothing

        # Serving link has already dropped below the outage threshold:
        # find the best alternative band NOW (no lookahead).
        best_band, best_snr = None, -99.0
        for bs in self.bss:
            if bs.band == ue.assigned_band:
                continue
            cand_snr = self.prop.snr_db(bs, ue)
            if cand_snr > best_snr:
                best_snr, best_band = cand_snr, bs.band

        if best_band is None:
            return None

        decision = ReactiveHandoffDecision(
            ue_id=ue.ue_id, trigger_time_ms=time_ms,
            source_band=ue.assigned_band, target_band=best_band,
            snr_at_trigger=snr,
        )
        self.handoff_log.append(decision)
        self._cooldown_remaining[ue.ue_id] = self.COOLDOWN_TICKS
        return decision

    def summary(self):
        print("\n" + "=" * 60)
        print("  REACTIVE HANDOFF ENGINE (Reactive-HO baseline) — SUMMARY")
        print("=" * 60)
        print(f"  Total HO decisions : {len(self.handoff_log)}")
        if self.handoff_log:
            snrs = [d.snr_at_trigger for d in self.handoff_log]
            print(f"  Avg SNR at trigger : {np.mean(snrs):.1f} dB "
                  f"(all below the {self.SNR_OUTAGE_DB} dB outage floor, by construction)")
        print("=" * 60)
