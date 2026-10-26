"""Tests for the PPE rule engine.

Runs standalone (`python test_ppe_rules.py`) or under pytest. No camera, no
model, no hardware, no numpy -- just synthetic boxes, which is the point: the
violation logic is the part worth pinning down precisely, and it should be
testable on any machine.

Frame is 640x480 throughout. A "near" person is ~300px tall (62% of frame),
which passes the view gate; a "far" person is ~100px tall and does not.
"""

from __future__ import annotations

import sys

from ppe_classes import CLASS_IDS
from ppe_config import ModelConfig, RulesConfig
from ppe_rules import RuleEngine, summarize
from ppe_types import Detection

FRAME = (640, 480)


def det(name: str, box: tuple[float, float, float, float], conf: float = 0.9) -> Detection:
    return Detection(cls_id=CLASS_IDS[name], name=name, conf=conf, xyxy=box)


def engine(required: list[str] | None = None, **rule_kw) -> RuleEngine:
    rules = RulesConfig(required_ppe=required or ["helmet"], **rule_kw)
    return RuleEngine(rules, ModelConfig())


# Geometry helpers. A person 300px tall centred in frame, plus a head box at
# the top of them, matching how the detector actually labels these.
NEAR_PERSON = (250.0, 100.0, 390.0, 400.0)   # 140x300, h_frac 0.625
NEAR_HEAD = (295.0, 110.0, 345.0, 170.0)     # inside the top 20% of the person
NEAR_HELMET = (292.0, 100.0, 348.0, 150.0)   # overlapping the head, slightly above
FAR_PERSON = (300.0, 300.0, 340.0, 400.0)    # 40x100, h_frac 0.208
FAR_HEAD = (310.0, 305.0, 330.0, 325.0)


def status(obs_list, item="helmet", index=0):
    return obs_list[index].items[item].status


# --------------------------------------------------------------------------
# the core case: helmet, positive evidence
# --------------------------------------------------------------------------


def test_bare_head_is_a_violation():
    """The whole product rests on this one."""
    obs = engine().evaluate_frame(
        [det("person", NEAR_PERSON, 0.91), det("head", NEAR_HEAD, 0.88)], FRAME
    )
    assert len(obs) == 1
    assert status(obs) == "MISSING_POSITIVE"
    assert "head" in obs[0].items["helmet"].support


def test_helmet_alone_is_compliant():
    obs = engine().evaluate_frame(
        [det("person", NEAR_PERSON, 0.91), det("helmet", NEAR_HELMET, 0.80)], FRAME
    )
    assert status(obs) == "PRESENT"


def test_head_and_helmet_together_is_compliant():
    """A helmeted person often yields BOTH boxes. Must read as compliant.

    This is also the case that would break if NMS were class-agnostic: one of
    the two overlapping boxes would have been suppressed.
    """
    obs = engine().evaluate_frame(
        [
            det("person", NEAR_PERSON, 0.91),
            det("head", NEAR_HEAD, 0.85),
            det("helmet", NEAR_HELMET, 0.78),
        ],
        FRAME,
    )
    assert status(obs) == "PRESENT"
    assert "covered by helmet" in obs[0].items["helmet"].reason


def test_carried_helmet_is_still_a_violation():
    """A helmet in the frame but not ON the detected head is not being worn.

    Someone holding a helmet, or one resting on a bench behind them, lands
    inside the person box. The naive reading -- "a helmet was detected,
    therefore compliant" -- is exactly the loophole this closes.
    """
    held_helmet = (255.0, 155.0, 295.0, 195.0)  # upper body, but clear of the head
    obs = engine().evaluate_frame(
        [
            det("person", NEAR_PERSON, 0.91),
            det("head", NEAR_HEAD, 0.88),
            det("helmet", held_helmet, 0.80),
        ],
        FRAME,
    )
    assert status(obs) == "MISSING_POSITIVE"
    assert "not covering it" in obs[0].items["helmet"].reason


def test_nothing_visible_is_unknown_not_a_violation():
    """No head, no helmet -> we don't know. Crying wolf here poisons the portal."""
    obs = engine().evaluate_frame([det("person", NEAR_PERSON, 0.91)], FRAME)
    assert status(obs) == "UNKNOWN"


def test_low_confidence_head_is_unknown():
    """Good enough to be a detection (>=0.50), not good enough to be proof (>=0.60)."""
    obs = engine().evaluate_frame(
        [det("person", NEAR_PERSON, 0.91), det("head", NEAR_HEAD, 0.55)], FRAME
    )
    assert status(obs) == "UNKNOWN"
    assert "below evidence bar" in obs[0].items["helmet"].reason


def test_presence_veto_beats_a_confident_bare_head():
    """A confident helmet clears the person even with a strong head box.

    Positive evidence of compliance should win; the alternative is accusing
    someone who is visibly wearing a helmet.
    """
    obs = engine().evaluate_frame(
        [
            det("person", NEAR_PERSON, 0.91),
            det("head", NEAR_HEAD, 0.99),
            det("helmet", NEAR_HELMET, 0.40),
        ],
        FRAME,
    )
    assert status(obs) == "PRESENT"


# --------------------------------------------------------------------------
# anatomy gate
# --------------------------------------------------------------------------


def test_helmet_at_knee_height_is_ignored():
    """A 'helmet' low on the body is a misdetection and must not clear anyone."""
    knee_helmet = (295.0, 330.0, 345.0, 380.0)  # ~77-93% down the person
    obs = engine().evaluate_frame(
        [
            det("person", NEAR_PERSON, 0.91),
            det("head", NEAR_HEAD, 0.88),
            det("helmet", knee_helmet, 0.85),
        ],
        FRAME,
    )
    assert status(obs) == "MISSING_POSITIVE", "knee-level helmet must not satisfy the rule"


def test_head_above_person_box_still_associates():
    """Detector person boxes routinely clip the top of the head."""
    high_head = (295.0, 80.0, 345.0, 125.0)  # starts 20px above the person box
    obs = engine().evaluate_frame(
        [det("person", NEAR_PERSON, 0.91), det("head", high_head, 0.88)], FRAME
    )
    assert status(obs) == "MISSING_POSITIVE"


# --------------------------------------------------------------------------
# multiple people
# --------------------------------------------------------------------------


def test_two_separate_people_judged_independently():
    left_person = (60.0, 100.0, 200.0, 400.0)
    left_head = (105.0, 110.0, 155.0, 170.0)
    obs = engine().evaluate_frame(
        [
            det("person", left_person, 0.90),
            det("head", left_head, 0.88),            # bare -> violation
            det("person", NEAR_PERSON, 0.91),
            det("helmet", NEAR_HELMET, 0.80),        # helmeted -> compliant
        ],
        FRAME,
    )
    assert len(obs) == 2
    by_x = sorted(obs, key=lambda o: o.person.x1)
    assert by_x[0].items["helmet"].status == "MISSING_POSITIVE"
    assert by_x[1].items["helmet"].status == "PRESENT"


def test_overlapping_people_with_one_helmet_is_unknown():
    """Ambiguous ownership must refuse to answer rather than guess.

    Two heavily-overlapping people and a single helmet: attributing it to
    either one invents a violation for the other.
    """
    a = (250.0, 100.0, 390.0, 400.0)
    b = (270.0, 100.0, 410.0, 400.0)  # overlaps a almost entirely
    shared_helmet = (320.0, 105.0, 360.0, 150.0)
    obs = engine().evaluate_frame(
        [det("person", a, 0.90), det("person", b, 0.90), det("helmet", shared_helmet, 0.85)],
        FRAME,
    )
    assert len(obs) == 2
    assert all(o.items["helmet"].status == "UNKNOWN" for o in obs)
    assert all("overlapping people" in o.items["helmet"].reason for o in obs)


# --------------------------------------------------------------------------
# view gating and absence-only items
# --------------------------------------------------------------------------


def test_distant_person_fails_the_view_gate():
    obs = engine().evaluate_frame(
        [det("person", FAR_PERSON, 0.91), det("head", FAR_HEAD, 0.88)], FRAME
    )
    assert obs[0].view_ok is False
    assert obs[0].height_frac < 0.35


def test_distant_bare_head_is_still_a_violation():
    """The view gate restricts ABSENCE evidence only.

    A clearly-visible bare head is proof regardless of how far away it is, so
    gating positive evidence on person size would throw away good detections.
    """
    obs = engine().evaluate_frame(
        [det("person", FAR_PERSON, 0.91), det("head", FAR_HEAD, 0.88)], FRAME
    )
    assert status(obs) == "MISSING_POSITIVE"


def test_vest_absence_needs_a_clear_view():
    obs = engine(["safety-vest"]).evaluate_frame(
        [det("person", FAR_PERSON, 0.91)], FRAME
    )
    assert status(obs, "safety-vest") == "UNKNOWN"
    assert "view inadequate" in obs[0].items["safety-vest"].reason


def test_vest_absence_on_a_clear_view_is_weak_missing():
    obs = engine(["safety-vest"]).evaluate_frame(
        [det("person", NEAR_PERSON, 0.91)], FRAME
    )
    assert status(obs, "safety-vest") == "MISSING_ABSENT"
    assert obs[0].items["safety-vest"].is_missing


def test_truncated_person_is_not_a_vest_violation():
    """Half out of frame: the vest may simply be outside the picture."""
    edge_person = (0.0, 100.0, 140.0, 400.0)  # flush against the left edge
    obs = engine(["safety-vest"]).evaluate_frame(
        [det("person", edge_person, 0.91)], FRAME
    )
    assert obs[0].truncated is True
    assert status(obs, "safety-vest") == "UNKNOWN"


def test_vest_suit_variants_satisfy_the_requirement():
    for alt in ("safety-vest", "safety-suit", "medical-suit"):
        obs = engine(["safety-vest"]).evaluate_frame(
            [
                det("person", NEAR_PERSON, 0.91),
                det(alt, (260.0, 170.0, 380.0, 330.0), 0.70),
            ],
            FRAME,
        )
        assert status(obs, "safety-vest") == "PRESENT", f"{alt} should satisfy vest"


# --------------------------------------------------------------------------
# multi-item, and non-person noise
# --------------------------------------------------------------------------


def test_helmet_and_gloves_assessed_separately():
    obs = engine(["helmet", "gloves"]).evaluate_frame(
        [
            det("person", NEAR_PERSON, 0.91),
            det("head", NEAR_HEAD, 0.88),                       # no helmet
            det("gloves", (255.0, 250.0, 295.0, 290.0), 0.70),  # gloved
        ],
        FRAME,
    )
    assert status(obs, "helmet") == "MISSING_POSITIVE"
    assert status(obs, "gloves") == "PRESENT"
    assert obs[0].missing_items() == ["helmet"]


def test_unrelated_classes_do_not_affect_verdicts():
    obs = engine().evaluate_frame(
        [
            det("person", NEAR_PERSON, 0.91),
            det("head", NEAR_HEAD, 0.88),
            det("tool", (255.0, 260.0, 300.0, 300.0), 0.80),
            det("shoes", (300.0, 375.0, 345.0, 398.0), 0.75),
        ],
        FRAME,
    )
    assert status(obs) == "MISSING_POSITIVE"


def test_detections_with_no_person_yield_nothing():
    obs = engine().evaluate_frame([det("head", NEAR_HEAD, 0.88)], FRAME)
    assert obs == []


def test_low_confidence_person_is_dropped():
    obs = engine().evaluate_frame(
        [det("person", NEAR_PERSON, 0.30), det("head", NEAR_HEAD, 0.88)], FRAME
    )
    assert obs == []


def test_empty_frame():
    assert engine().evaluate_frame([], FRAME) == []
    assert summarize([]) == "no people detected"


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures: list[tuple[str, BaseException]] = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except BaseException as exc:  # noqa: BLE001 - test runner
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("\nfailures:")
        for name, exc in failures:
            print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    print("PPE rule engine tests\n")
    sys.exit(_run())
