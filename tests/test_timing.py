from aivoice.timing import duration_tolerance_seconds, evaluate_duration_fit


def test_duration_tolerance_uses_ratio_for_short_cues() -> None:
    assert duration_tolerance_seconds(2.0) == 0.16


def test_duration_tolerance_is_capped_for_long_cues() -> None:
    assert duration_tolerance_seconds(8.0) == 0.4


def test_duration_fit_accepts_within_policy() -> None:
    fit = evaluate_duration_fit(target_duration=4.0, measured_duration=4.25)

    assert fit.within_tolerance
    assert fit.action == "accept"
    assert fit.tolerance_seconds == 0.32


def test_duration_fit_routes_large_errors_to_hard_duration_or_segmentation() -> None:
    assert evaluate_duration_fit(4.0, 4.7).action == "retry_or_hard_duration"
    assert evaluate_duration_fit(4.0, 5.2).action == "split_or_merge"
