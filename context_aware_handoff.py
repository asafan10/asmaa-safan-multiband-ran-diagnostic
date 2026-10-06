"""
=============================================================================
  MODULE 2 — Context-Aware Band Selection & Handoff Decision Process
=============================================================================
Features:
  • Context vector (QoS, mobility, application type, battery, interference)
  • Multi-criteria decision making (TOPSIS-based scoring)
  • Hysteresis + time-to-trigger handoff filter (A3 / A5 events)
  • Event-driven handoff state machine
=============================================================================

  >>> Weighting design note <<<
  TOPSISBandScorer supports TWO weighting modes:

    - "static" (default, matches the original paper): a single criterion
      weight vector [throughput, latency, SNR, battery, HO cost] =
      [0.35, 0.25, 0.20, 0.10, 0.10] applied to every UE regardless of
      its application type. Per-application QoS is instead enforced
      *upstream* via the feasibility gate (Sec. III.B.1) using
      QOS_PROFILES — an app that needs sub-5ms latency (VR/XR) simply
      has bands that can't meet it marked infeasible (score forced to 0)
      rather than the TOPSIS weights themselves shifting per app.

    - "app_aware": weights are drawn from WEIGHTS_BY_APP below, so e.g.
      Voice/VoNR weights latency far more heavily than a Bulk Data
      transfer does, instead of relying purely on the feasibility gate.

  The previous version of this file only implemented "static" while
  simultaneously computing a per-UE `ctx.app_type` that had no effect
  on the weights at all — the AppType was measured but never used
  downstream. Both modes are now real, selectable via
  `TOPSISBandScorer(weight_mode="static" | "app_aware")`, so this is an
  explicit, testable design decision rather than a silent gap. Run
  run_experiments.py's `compare_weight_modes()` to reproduce the
  ablation between the two.
=============================================================================
"""

import time
import math
import random
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from multiband_planning import (
    Band, BaseStation, UserEquipment, PropagationModel, BANDWIDTH_HZ,
    LATENCY_MS, PEAK_TPUT_MBPS, HO_PENALTY_MS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Context & QoS primitives
# ─────────────────────────────────────────────────────────────────────────────

class AppType(Enum):
    VR_XR        = "VR/XR"            # Ultra-high throughput, low latency
    VIDEO_STREAM = "Video Streaming"  # High throughput, moderate latency
    VOICE        = "Voice/VoNR"       # Low throughput, very low latency
    BULK_DATA    = "Bulk Data"        # High throughput, latency tolerant
    IOT          = "IoT/mMTC"         # Very low throughput, low power


@dataclass
class QoSRequirement:
    min_throughput_mbps: float
    max_latency_ms:      float
    max_bler:            float       # Block Error Rate
    priority:            int         # 1 (highest) – 9 (lowest)


QOS_PROFILES: Dict[AppType, QoSRequirement] = {
    AppType.VR_XR:        QoSRequirement(500,  5,    0.01, 1),
    AppType.VIDEO_STREAM: QoSRequirement(50,   50,   0.02, 3),
    AppType.VOICE:        QoSRequirement(0.5,  20,   0.01, 2),
    AppType.BULK_DATA:    QoSRequirement(100,  500,  0.05, 6),
    AppType.IOT:          QoSRequirement(0.1,  1000, 0.10, 9),
}

# Per-application TOPSIS weight vectors [throughput, latency, SNR, battery, HO cost].
# Each row still sums to 1.0. Derived from QOS_PROFILES priorities: apps with a
# tight max_latency_ms get more latency weight; IoT (power-constrained) gets
# more battery weight; bulk/video (throughput-bound) get more throughput weight.
WEIGHTS_BY_APP: Dict[AppType, np.ndarray] = {
    AppType.VR_XR:        np.array([0.30, 0.35, 0.20, 0.05, 0.10]),
    AppType.VIDEO_STREAM: np.array([0.45, 0.20, 0.20, 0.05, 0.10]),
    AppType.VOICE:        np.array([0.15, 0.45, 0.20, 0.05, 0.15]),
    AppType.BULK_DATA:    np.array([0.55, 0.05, 0.20, 0.10, 0.10]),
    AppType.IOT:          np.array([0.15, 0.15, 0.20, 0.40, 0.10]),
}
STATIC_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.10, 0.10])


@dataclass
class UEContext:
    ue_id:           int
    app_type:        AppType
    velocity_mps:    float       # m/s
    battery_pct:     float       # 0-100
    snr_db:          Dict[Band, float] = field(default_factory=dict)
    interference_db: Dict[Band, float] = field(default_factory=dict)
    rsrp_dbm:        Dict[Band, float] = field(default_factory=dict)
    serving_band:    Optional[Band]    = None
    timestamp:       float             = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Band capability model
# ─────────────────────────────────────────────────────────────────────────────
# LATENCY_MS, PEAK_TPUT_MBPS, and HO_PENALTY_MS are imported from
# multiband_planning (single source of truth, shared with Module 4's
# reward shaping).

# Power consumption weight (higher -> more drain)
POWER_WEIGHT = {Band.SUB6: 1.0, Band.MMWAVE: 2.5, Band.THZ: 5.0}


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  TOPSIS multi-criteria scorer
# ─────────────────────────────────────────────────────────────────────────────

class TOPSISBandScorer:
    """
    Ranks bands using Technique for Order of Preference by
    Similarity to Ideal Solution (TOPSIS).

    Criteria (all normalised to [0, 1] internally):
      c1: Estimated throughput  (benefit, higher is better)
      c2: Latency               (cost, lower is better -> inverted)
      c3: SNR                   (benefit, higher is better)
      c4: Battery impact        (cost, lower is better -> inverted)
      c5: Handoff cost          (cost, lower is better -> inverted)

    weight_mode: "static" (paper default) or "app_aware" (see module
    docstring for the design rationale behind offering both).
    """

    def __init__(self, weight_mode: str = "static"):
        if weight_mode not in ("static", "app_aware"):
            raise ValueError("weight_mode must be 'static' or 'app_aware'")
        self.weight_mode = weight_mode

    def _weights_for(self, app_type: AppType) -> np.ndarray:
        if self.weight_mode == "app_aware":
            return WEIGHTS_BY_APP[app_type]
        return STATIC_WEIGHTS

    @staticmethod
    def _estimate_throughput(band: Band, snr_db: float) -> float:
        """Shannon-inspired normalised throughput estimate."""
        bw = BANDWIDTH_HZ[band]
        snr_lin = max(10 ** (snr_db / 10), 1e-3)
        raw_mbps = 0.65 * bw * math.log2(1 + snr_lin) / 1e6
        return min(raw_mbps, PEAK_TPUT_MBPS[band])

    def score(
        self,
        ctx: UEContext,
        candidate_bands: List[Band],
        prop_model: PropagationModel,
        serving_band: Optional[Band],
    ) -> Dict[Band, float]:
        """Returns a dict {band: score in [0,1]}, sorted descending."""
        if not candidate_bands:
            return {}

        qos = QOS_PROFILES[ctx.app_type]

        # Build criteria matrix [n_bands x 5]
        matrix = []
        for band in candidate_bands:
            snr  = ctx.snr_db.get(band, -20)
            tput = self._estimate_throughput(band, snr)
            lat  = LATENCY_MS[band]
            bat  = POWER_WEIGHT[band] * (1.0 - ctx.battery_pct / 100.0)
            ho   = HO_PENALTY_MS.get((serving_band, band), 0) if serving_band and serving_band != band else 0
            matrix.append([tput, lat, snr, bat, ho])

        matrix = np.array(matrix, dtype=float)
        n_bands, n_crit = matrix.shape

        # --- TOPSIS Steps ---
        # Step 1: Vector-normalise each criterion column
        col_norm = np.linalg.norm(matrix, axis=0)
        col_norm[col_norm == 0] = 1
        norm_matrix = matrix / col_norm

        # Step 2: Weighted normalised matrix
        w = self._weights_for(ctx.app_type)[:n_crit]
        w = w / w.sum()
        weighted = norm_matrix * w

        # Step 3: Ideal best & worst
        # Criteria types: benefit=[throughput, SNR], cost=[latency, battery, HO]
        benefit_idx = [0, 2]
        cost_idx    = [1, 3, 4]
        ideal_best  = weighted.max(axis=0).copy()
        ideal_worst = weighted.min(axis=0).copy()
        for idx in cost_idx:
            ideal_best[idx], ideal_worst[idx] = weighted[:, idx].min(), weighted[:, idx].max()

        # Step 4: Euclidean distances to ideal best/worst
        d_best  = np.linalg.norm(weighted - ideal_best,  axis=1)
        d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)

        # Step 5: Relative closeness C* in [0, 1]
        denom = d_best + d_worst
        denom[denom == 0] = 1e-9
        scores = d_worst / denom

        # Apply QoS feasibility gate — a band that cannot physically meet
        # this application's minimum throughput / max latency is scored 0
        # regardless of its TOPSIS closeness.
        qos_scores = {}
        for i, band in enumerate(candidate_bands):
            tput_est = matrix[i, 0]
            lat_est  = matrix[i, 1]
            feasible = (tput_est >= qos.min_throughput_mbps * 0.8 and
                        lat_est  <= qos.max_latency_ms)
            qos_scores[band] = float(scores[i]) if feasible else 0.0

        return dict(sorted(qos_scores.items(), key=lambda x: -x[1]))


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Handoff state machine (A3/A5 events + hysteresis + TTT)
# ─────────────────────────────────────────────────────────────────────────────

class HandoffEvent(Enum):
    A3 = "A3"   # Neighbour becomes offset-better than serving
    A5 = "A5"   # Serving falls below T1 AND neighbour exceeds T2


@dataclass
class HandoffTrigger:
    event:         HandoffEvent
    source_band:   Band
    target_band:   Band
    trigger_time:  float = field(default_factory=time.time)
    ttt_ms:        float = 40.0    # Time-To-Trigger (ms)
    hysteresis_db: float = 3.0     # dB


class HandoffState(Enum):
    IDLE      = auto()
    MEASURING = auto()
    TRIGGERED = auto()
    EXECUTING = auto()
    COMPLETE  = auto()


class HandoffStateMachine:
    """3GPP-like handoff state machine with hysteresis and TTT filtering."""

    # A5 thresholds
    A5_THRESHOLD1_DB = 0   # Serving RSRP must drop below this
    A5_THRESHOLD2_DB = 5   # Neighbour RSRP must rise above this

    # Time-to-trigger per (source, target) band pair (ms)
    TTT_MS = {
        (Band.SUB6,   Band.MMWAVE): 40,
        (Band.SUB6,   Band.THZ):    20,
        (Band.MMWAVE, Band.THZ):    10,
        (Band.MMWAVE, Band.SUB6):   80,
        (Band.THZ,    Band.MMWAVE): 30,
        (Band.THZ,    Band.SUB6):   60,
    }

    def __init__(self, ue_id: int):
        self.ue_id            = ue_id
        self.state             = HandoffState.IDLE
        self.pending_trigger:  Optional[HandoffTrigger] = None
        self.trigger_start:    Optional[float]          = None
        self.history:          List[Dict]               = []

    def evaluate(
        self,
        ctx: UEContext,
        scores: Dict[Band, float],
        current_time_ms: float = 0.0,
    ) -> Optional[Tuple[Band, Band]]:
        """Returns (source_band, target_band) if a handoff should execute, else None."""
        if not ctx.serving_band or not scores:
            return None

        best_band  = next(iter(scores))
        best_score = scores[best_band]
        curr_score = scores.get(ctx.serving_band, 0.0)

        if best_band == ctx.serving_band:
            self.state = HandoffState.IDLE
            self.pending_trigger = None
            return None

        snr_diff = (ctx.snr_db.get(best_band, -99) -
                    ctx.snr_db.get(ctx.serving_band, -99))
        a3_fired = snr_diff > 3.0 and best_score > curr_score + 0.05

        a5_fired = (ctx.snr_db.get(ctx.serving_band, 0) < self.A5_THRESHOLD1_DB and
                    ctx.snr_db.get(best_band, -99)        > self.A5_THRESHOLD2_DB)

        if a3_fired or a5_fired:
            event = HandoffEvent.A3 if a3_fired else HandoffEvent.A5
            if self.state == HandoffState.IDLE:
                ttt = self.TTT_MS.get((ctx.serving_band, best_band), 40)
                self.pending_trigger = HandoffTrigger(
                    event=event, source_band=ctx.serving_band, target_band=best_band,
                    trigger_time=current_time_ms, ttt_ms=ttt,
                )
                self.trigger_start = current_time_ms
                self.state = HandoffState.TRIGGERED

            elif self.state == HandoffState.TRIGGERED:
                elapsed = current_time_ms - (self.trigger_start or 0)
                ttt_needed = self.pending_trigger.ttt_ms if self.pending_trigger else 40
                if elapsed >= ttt_needed:
                    src = self.pending_trigger.source_band
                    tgt = self.pending_trigger.target_band
                    self.history.append({
                        "time_ms": current_time_ms, "event": self.pending_trigger.event.value,
                        "source": src.value, "target": tgt.value,
                        "snr_diff_db": round(snr_diff, 2),
                    })
                    # Reset straight back to IDLE rather than leaving state
                    # at COMPLETE/EXECUTING: those aren't states this FSM
                    # ever transitions OUT of again (the if/elif above only
                    # matches IDLE or TRIGGERED), so leaving state at
                    # COMPLETE would permanently strand this UE — it could
                    # never detect or execute another handoff afterward.
                    # One decision fully resolves in one evaluate() call;
                    # there's nothing to keep "in progress" here.
                    self.state = HandoffState.IDLE
                    self.pending_trigger = None
                    return (src, tgt)
        else:
            if self.state == HandoffState.TRIGGERED:
                self.state = HandoffState.IDLE
                self.pending_trigger = None

        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Context-aware band selector (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class ContextAwareBandSelector:
    """Orchestrates context collection -> TOPSIS scoring -> handoff decisions."""

    def __init__(self, weight_mode: str = "static"):
        self.scorer  = TOPSISBandScorer(weight_mode=weight_mode)
        self.prop    = PropagationModel()
        self.ho_fsm: Dict[int, HandoffStateMachine] = {}

    def _get_or_create_fsm(self, ue_id: int) -> HandoffStateMachine:
        if ue_id not in self.ho_fsm:
            self.ho_fsm[ue_id] = HandoffStateMachine(ue_id)
        return self.ho_fsm[ue_id]

    def build_context(
        self,
        ue: UserEquipment,
        app_type: AppType,
        bss: List[BaseStation],
    ) -> UEContext:
        """Measure SNR / RSRP from all visible BSs and populate context."""
        ctx = UEContext(
            ue_id        = ue.ue_id,
            app_type     = app_type,
            velocity_mps = ue.velocity,
            battery_pct  = random.uniform(20, 100),
            serving_band = ue.assigned_band,
        )
        for band in Band:
            band_bss = [b for b in bss if b.band == band]
            if not band_bss:
                continue
            snrs = [self.prop.snr_db(b, ue) for b in band_bss]
            ctx.snr_db[band]   = max(snrs)
            ctx.rsrp_dbm[band] = max(self.prop.received_power_dbm(b, ue) for b in band_bss)
            ctx.interference_db[band] = float(np.percentile(snrs, 25)) if len(snrs) > 1 else -30
        return ctx

    def select_band(
        self,
        ctx: UEContext,
        time_ms: float = 0.0,
    ) -> Tuple[Optional[Band], Optional[Tuple[Band, Band]]]:
        """
        Returns (selected_band, handoff_event_or_None).

        >>> Fixed: TOPSIS vs. A3/A5 conflict was silently unresolved <<<
        A technical review asked directly: "what happens when TOPSIS and
        the A3/A5 hysteresis FSM disagree?" The honest answer, before this
        fix, was: the FSM's decision was computed and its internal state
        updated, but then completely discarded — this method always
        returned TOPSIS's raw top-ranked band regardless of what the FSM
        said, on every single call. That makes the entire hysteresis/TTT
        mechanism (the whole point of HandoffStateMachine — don't switch
        bands on every noisy SNR sample) decorative: a caller reading the
        first return value would see the UE "selected" a different band
        on every call as soon as TOPSIS's ranking flickered, with no
        hysteresis protection at all, even though `ho_event` (the second
        return value) correctly stayed None until the FSM's TTT window
        elapsed.

        Fixed behavior: the FSM wins. If the FSM has not authorized a
        handoff this call (ho_event is None), the UE STAYS on its current
        serving band — even if TOPSIS's raw ranking currently prefers
        something else — exactly mirroring real 3GPP behavior where a
        UE doesn't re-associate until its measurement report actually
        triggers and clears time-to-trigger. Only when the FSM fires
        (ho_event is not None) does the returned band change.

        A UE with no serving band yet (first assignment, ctx.serving_band
        is None) has no "current band" for hysteresis to protect, so it
        bootstraps directly to TOPSIS's top-ranked pick.
        """
        candidates = [b for b in Band if b in ctx.snr_db]
        scores = self.scorer.score(ctx, candidates, self.prop, ctx.serving_band)

        if not scores:
            return ctx.serving_band, None

        fsm = self._get_or_create_fsm(ctx.ue_id)
        ho_event = fsm.evaluate(ctx, scores, time_ms)

        if ho_event is not None:
            _, target_band = ho_event
            return target_band, ho_event

        if ctx.serving_band is not None:
            return ctx.serving_band, None   # FSM didn't authorize a switch — stay put

        best_band = next(iter(scores))       # no serving band yet — bootstrap
        return best_band, None

    def print_decision(self, ctx: UEContext, band: Band, ho_event):
        print(f"\n  UE {ctx.ue_id} | App: {ctx.app_type.value} | "
              f"vel={ctx.velocity_mps:.1f} m/s | bat={ctx.battery_pct:.0f}%")
        print(f"    SNR  : " + " | ".join(
            f"{b.value}: {ctx.snr_db.get(b, -99):.1f} dB" for b in Band))
        print(f"    -> Selected band : {band.value}")
        if ho_event:
            print(f"    HANDOFF: {ho_event[0].value} -> {ho_event[1].value}")
        else:
            print(f"    No handoff required")


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  Baseline BL-2 — simple threshold-based band selection (no TOPSIS)
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdBandSelector:
    """
    Baseline (BL-2), requested explicitly by the technical review as a
    "simple threshold-based band selection" comparison point distinct
    from both BL-1 (Max-SNR) and the proposed TOPSIS scorer.

    Rule-based, no multi-criteria optimization: upgrade to the next
    higher band ONLY if its SNR clears both (a) the band's own minimum
    usability threshold and (b) a fixed upgrade margin over the
    currently serving band's SNR — otherwise stay put. No QoS profiles,
    no weights, no learning. This is the kind of naive rule an operator
    might ship before investing in anything smarter, and is a fairer
    comparison for "is TOPSIS worth it?" than Max-SNR is, since it at
    least has hysteresis (unlike BL-1) but no per-application awareness
    or multi-criteria trade-off (unlike the proposed Module 2).
    """

    UPGRADE_MARGIN_DB = 5.0    # must beat current band by this much to switch
    BAND_ORDER = [Band.SUB6, Band.MMWAVE, Band.THZ]   # low -> high capability

    def __init__(self):
        self.prop = PropagationModel()

    def select_band(self, ue: UserEquipment, bss: List[BaseStation]) -> Optional[Band]:
        snr_by_band: Dict[Band, float] = {}
        for band in self.BAND_ORDER:
            band_bss = [b for b in bss if b.band == band]
            snr_by_band[band] = max((self.prop.snr_db(b, ue) for b in band_bss),
                                     default=-99.0)

        USABLE_FLOOR_DB = -90.0   # "no real signal" floor
        current = ue.assigned_band or Band.SUB6
        current_idx = self.BAND_ORDER.index(current)

        best = current
        for band in self.BAND_ORDER[current_idx + 1:]:
            if (snr_by_band[band] > USABLE_FLOOR_DB and
                    snr_by_band[band] >= snr_by_band[current] + self.UPGRADE_MARGIN_DB):
                best = band

        if snr_by_band[best] > USABLE_FLOOR_DB:
            return best
        if snr_by_band[current] > USABLE_FLOOR_DB:
            return current
        return None
