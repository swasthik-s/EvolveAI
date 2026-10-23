import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn"))
from autonomous_lab import propose_trials


def test_trial_search_is_bounded_and_deterministic():
    first = propose_trials(4, seed=17)
    second = propose_trials(4, seed=17)
    assert first == second
    assert len(first) == 4
    assert {trial.architecture for trial in first} <= {"baseline", "evolve_context"}
    assert all(trial.hidden_size in {32, 64, 96, 128} for trial in first)
    assert all(trial.learning_rate in {0.001, 0.0005} for trial in first)


def test_trial_count_is_clamped_to_safe_range():
    assert len(propose_trials(0)) == 1
    assert len(propose_trials(100)) == 6
