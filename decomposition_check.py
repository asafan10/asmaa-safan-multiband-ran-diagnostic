"""
Controlled decomposition of the gap between Table V.a's one-shot snapshot for Module 1 (890.41 Mbps) and the
150-tick figure of Table V.c/V.d and Table VII row A1 (355.85 Mbps), same 20 seeds, same K=150 population.

  (a) snapshot               : throughput read once, right after placement        (Table V.a)
  (b) tick-redraw, no motion : A1 harness, UEs not moved, shadow fading redrawn every tick
  (c) motion, frozen shadow  : A1 harness, UEs move, but each (UE, BS) shadow-fading draw is the one made at
                               placement and is then frozen (a first-call-wins cache)
  (d) motion + tick-redraw   : A1 harness unchanged                                (Table VII row A1)

Usage: python3 decomposition_check.py   (a few minutes on a laptop; NumPy only)
"""
import json, random
import numpy as np
import ablation_experiment as ab
from multiband_planning import PropagationModel, compute_network_kpis

SEEDS = ab.SEEDS
_orig = PropagationModel.received_power_dbm.__func__
_cache = {}
_frozen = False

def _patched(cls, bs, ue):
    if not _frozen:
        return _orig(cls, bs, ue)
    key = (bs.bs_id, ue.ue_id)
    if key not in _cache:
        _cache[key] = _orig(cls, bs, ue)
    return _cache[key]

def run(seed, move, frozen):
    global _frozen, _cache
    _cache = {}; _frozen = frozen
    planner, ues, _ = ab._build_population(seed)     # planning draws happen here (cached if frozen)
    rng = np.random.default_rng(seed + 10_000)
    snaps = []
    for tick in range(ab.N_TICKS):
        for ue in ues:
            if move: ab._move(ue, rng)
            ab._recompute_throughput_frozen(planner, ue)
        snaps.append(ab._snapshot_kpis(planner, ues))
    _frozen = False
    return ab._avg_kpis(snaps)

PropagationModel.received_power_dbm = classmethod(_patched)

if __name__ == "__main__":
    out = {}
    for name, move, frozen in [("(b) tick-redraw, no motion", False, False),
                               ("(c) motion, frozen shadow", True, True),
                               ("(d) motion + tick-redraw (Table VII A1)", True, False)]:
        vals = [run(s, move, frozen)["avg_throughput_mbps"] for s in SEEDS]
        out[name] = vals
        print(f"{name:45s} mean avg throughput = {np.mean(vals):8.2f} Mbps (SD {np.std(vals):.2f})", flush=True)
    json.dump(out, open("decomposition_check_results.json", "w"), indent=1)
