"""
=============================================================================
  MAIN — Full Integration Demo
  Runs all four proposed modules end-to-end, plus the BL-1 / Reactive-HO baseline
  baselines from Section IV, in one coherent simulation scenario.
=============================================================================

This file is a single-seed, human-readable WALKTHROUGH of the pipeline —
good for sanity-checking behavior and for the console output the paper's
figures are illustrated from. It is NOT the statistical-comparison
harness: for the multi-seed, paired-t-test comparison against baselines
and the reward-weight / TOPSIS-weight ablations, see run_experiments.py.
=============================================================================
"""

import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Module imports ──────────────────────────────────────────────────────────
from multiband_planning import (
    HierarchicalMultiBandPlanner, MaxSNRPlanner, UserEquipment, Position, Band,
    build_scenario, clone_topology,
)
from context_aware_handoff import (
    ContextAwareBandSelector, AppType
)
from predictive_handoff import (
    PredictiveHandoffEngine, ReactiveHandoffEngine, Obstacle
)
from dqn_traffic_steering import (
    DQNTrainer, EnvConfig, TrainingConfig, RewardWeights
)


def run_demo():
    print("\n" + "#" * 65)
    print("  HIERARCHICAL MULTI-BAND RAN SIMULATION")
    print("  Sub-6 GHz | mmWave | THz  —  Full Pipeline Demo")
    print(f"  seed = {SEED}")
    print("#" * 65)

    # ── Step 1: Hierarchical network planning (proposed) vs. Max-SNR (BL-1) ──
    print("\n[1/5] Hierarchical Multi-Band Network Planning (proposed) ...")
    base_stations, ues = build_scenario(seed=SEED)

    planner = HierarchicalMultiBandPlanner()
    planner.base_stations = base_stations
    planner.ues = ues
    planner._bs_index = {b.bs_id: b for b in base_stations}
    planner.run_planning()
    planner.print_report()

    print("\n[1b/5] Baseline BL-1 — Max-SNR planner on the SAME topology ...")
    bl1_bss, bl1_ues = clone_topology(base_stations, ues)
    bl1 = MaxSNRPlanner(bl1_bss)
    bl1.ues = bl1_ues
    bl1.run_planning()
    bl1.print_report()

    # ── Step 2: Context-aware band selection ──────────────────────────────────
    print("\n[2/5] Context-Aware Band Selection & Handoff (TOPSIS) ...")
    selector = ContextAwareBandSelector(weight_mode="static")

    app_types = list(AppType)
    print("\n  Showing decisions for 5 sample UEs:")
    for ue in ues[:5]:
        app = random.choice(app_types)
        ctx = selector.build_context(ue, app, planner.base_stations)
        band, ho_event = selector.select_band(ctx, time_ms=100.0)
        selector.print_decision(ctx, band, ho_event)

    # ── Step 3: Predictive handoff (proposed) vs. reactive-only (Reactive-HO baseline) ────────
    print("\n[3/5] Predictive Handoff with Blockage Detection vs. Reactive Baseline ...")
    engine   = PredictiveHandoffEngine(planner.base_stations)
    reactive = ReactiveHandoffEngine(planner.base_stations)

    for ue in ues[:10]:
        engine.register_ue(ue)
        reactive.register_ue(ue)

    engine.add_obstacle(Obstacle(x_min=150, x_max=200, y_min=150, y_max=300, attenuation_db=30))
    engine.add_obstacle(Obstacle(x_min=300, x_max=350, y_min=50,  y_max=150, attenuation_db=25))
    engine.add_obstacle(Obstacle(x_min=50,  x_max=100, y_min=350, y_max=450, attenuation_db=20))

    print("\n  Simulating 20 ticks (100 ms each):")
    predictive_ho_count, reactive_ho_count = 0, 0
    for tick in range(20):
        for ue in ues[:10]:
            ue.position.x = max(0, min(500, ue.position.x + ue.velocity * 0.1 * random.uniform(-1, 1)))
            ue.position.y = max(0, min(500, ue.position.y + ue.velocity * 0.1 * random.uniform(-1, 1)))

            decision = engine.tick(ue, time_ms=tick * 100.0)
            if decision:
                engine.print_decision(decision)
                predictive_ho_count += 1

            if reactive.tick(ue, time_ms=tick * 100.0):
                reactive_ho_count += 1

    print(f"\n  Total proactive HOs triggered (proposed) : {predictive_ho_count}")
    print(f"  Total reactive HOs triggered  (Reactive-HO baseline)     : {reactive_ho_count}")
    engine.summary()
    reactive.summary()

    # ── Step 4: DQN training ──────────────────────────────────────────────────
    print("\n[4/5] Deep Q-Network Training for Traffic Steering ...")
    trainer = DQNTrainer(
        env_cfg=EnvConfig(n_ues=10, max_steps=50, reward_weights=RewardWeights()),
        train_cfg=TrainingConfig(n_episodes=300, log_interval=20, eval_interval=40),
        agent_kwargs=dict(lr=1e-4, gamma=0.99, eps_start=1.0, eps_end=0.05,
                           eps_decay=0.9999, batch_size=64, hidden=128),
    )
    trainer.train()
    trainer.plot_training_curves()
    trainer.band_selection_stats()

    print("\n[5/5] For statistical comparison against baselines across multiple")
    print("      seeds (paired t-tests), timing benchmarks, and the reward /")
    print("      TOPSIS weight ablations, run: python run_experiments.py")

    print("\n" + "#" * 65)
    print("  SIMULATION COMPLETE")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
