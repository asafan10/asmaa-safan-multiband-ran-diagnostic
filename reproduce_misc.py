"""Supporting results quoted in Sections V-B, V-D and V-F.
 - topsis_sensitivity            -> Fig. 5 numbers (mean rank stability 0.966, minimum 0.502, CV 3.75 %)
 - compare_position_trackers     -> Kalman vs. exponential smoothing (-8.7 %, p = 0.018, d = -0.58)
 - evaluate_blockage_prediction  -> 2 true positives of 21 genuine events, 17 firings
 - benchmark_timing              -> Section V-F, on the NumPy-MLP DQN backend (PyTorch disabled), 20 UEs
These functions print their results; compare with the paper text."""
import dqn_traffic_steering as dts
import run_experiments as r
print("\n##### topsis_sensitivity #####"); r.topsis_sensitivity()
print("\n##### compare_position_trackers #####"); r.compare_position_trackers()
print("\n##### evaluate_blockage_prediction (guaranteed crossings) #####"); r.evaluate_blockage_prediction(guarantee_blockage_crossings=True)
dts.TORCH_AVAILABLE = False
print("\n##### benchmark_timing (NumPy MLP backend, 20 UEs) #####"); r.benchmark_timing(n_ues=20)
