"""Graduation eligibility checks."""

from engine.gatekeeper import GraduationGatekeeper


def test_gatekeeper_requires_all_configured_thresholds():
    gate = GraduationGatekeeper(100_000)
    eligible = gate.evaluate(115_000, 30, 50.0, 5.0)
    assert eligible.unlocked
    assert not eligible.reasons

    blocked = gate.evaluate(114_000, 29, 49.0, 5.1)
    assert not blocked.unlocked
    assert len(blocked.reasons) == 4
