"""Tests for tracking, temporal confirmation and de-duplication.

Deterministic: time is injected, never read from the clock, so these cover the
cooldown and rate-limit logic without a single sleep.

Run: python test_ppe_tracker.py
"""

from __future__ import annotations

import sys

from ppe_classes import CLASS_IDS
from ppe_config import CooldownConfig, ModelConfig, RulesConfig, TemporalConfig
from ppe_rules import RuleEngine
from ppe_tracker import PatrolVisionState, PersonTracker, ViolationConfirmer
from ppe_types import Detection

FRAME = (640, 480)

NEAR_PERSON = (250.0, 100.0, 390.0, 400.0)
NEAR_HEAD = (295.0, 110.0, 345.0, 170.0)
NEAR_HELMET = (292.0, 100.0, 348.0, 150.0)


def det(name, box, conf=0.9):
    return Detection(cls_id=CLASS_IDS[name], name=name, conf=conf, xyxy=box)


def rules_engine(required=None):
    return RuleEngine(RulesConfig(required_ppe=required or ["helmet"]), ModelConfig())


def bare_head_frame(person=NEAR_PERSON, head=NEAR_HEAD):
    return [det("person", person, 0.91), det("head", head, 0.88)]


def helmeted_frame(person=NEAR_PERSON):
    return [det("person", person, 0.91), det("helmet", NEAR_HELMET, 0.80)]


def observe(dets, required=None):
    return rules_engine(required).evaluate_frame(dets, FRAME)


def temporal(**kw):
    base = dict(window_m=10, positive_n=5, absence_n=8, min_track_hits=4)
    base.update(kw)
    return TemporalConfig(**base)


def cooldown(**kw):
    base = dict(
        per_person_s=90.0,
        global_min_interval_s=8.0,
        max_per_hour=60,
        reid_similarity=0.85,
        reid_window_s=120.0,
    )
    base.update(kw)
    return CooldownConfig(**base)


def state(temporal_kw=None, cooldown_kw=None):
    return PatrolVisionState(
        temporal(**(temporal_kw or {})), cooldown(**(cooldown_kw or {}))
    )


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def test_same_person_keeps_one_track_id():
    tracker = PersonTracker()
    ids = set()
    for i in range(6):
        drift = (250.0 + i * 3, 100.0, 390.0 + i * 3, 400.0)
        obs = observe(bare_head_frame(person=drift))
        pairs = tracker.update(obs, now=float(i))
        ids.add(pairs[0][0].track_id)
    assert ids == {1}, f"a slowly-walking person must stay one track, got {ids}"


def test_two_people_get_two_ids():
    tracker = PersonTracker()
    far = (40.0, 100.0, 180.0, 400.0)
    obs = observe(bare_head_frame() + [det("person", far, 0.90)])
    pairs = tracker.update(obs, now=0.0)
    assert len({t.track_id for t, _ in pairs}) == 2


def test_track_ages_out_after_max_misses():
    tracker = PersonTracker(max_misses=3)
    tracker.update(observe(bare_head_frame()), now=0.0)
    assert len(tracker.tracks) == 1
    for i in range(4):
        tracker.update([], now=float(i + 1))
    assert tracker.tracks == {}


def test_track_survives_a_brief_miss():
    tracker = PersonTracker(max_misses=3)
    tracker.update(observe(bare_head_frame()), now=0.0)
    tracker.update([], now=1.0)
    pairs = tracker.update(observe(bare_head_frame()), now=2.0)
    assert pairs[0][0].track_id == 1
    assert pairs[0][0].misses == 0


def test_fast_motion_matches_by_centroid():
    """A jump too big for IoU should still match, not spawn a new id."""
    tracker = PersonTracker()
    tracker.update(observe(bare_head_frame()), now=0.0)
    moved = (310.0, 100.0, 450.0, 400.0)  # shifted 60px, IoU ~0.44 -> still ok
    far_jump = (355.0, 100.0, 495.0, 400.0)  # shifted 105px, IoU below thresh
    tracker.update(observe(bare_head_frame(person=moved)), now=1.0)
    pairs = tracker.update(observe(bare_head_frame(person=far_jump)), now=2.0)
    assert pairs[0][0].track_id == 1, "centroid fallback should hold identity"


# --------------------------------------------------------------------------
# N-of-M confirmation
# --------------------------------------------------------------------------


def test_single_bad_frame_does_not_fire():
    """The core anti-false-positive guarantee."""
    st = state()
    events = st.process(observe(bare_head_frame()), now=0.0)
    assert events == []


def test_fires_after_enough_agreeing_frames():
    st = state()
    fired = None
    for i in range(10):
        events = st.process(observe(bare_head_frame()), now=float(i))
        if events:
            fired = (i, events[0])
            break
    assert fired is not None, "5-of-10 agreement should eventually confirm"
    i, event = fired
    assert i == 4, f"expected confirmation on the 5th frame, got the {i + 1}th"
    assert event.items == ["helmet"]
    assert event.frames_agree["helmet"] == "5/5"
    assert event.track_id == 1


def test_min_track_hits_gates_confirmation():
    """Even unanimous frames wait for the track to establish itself."""
    st = state(temporal_kw={"positive_n": 2, "min_track_hits": 6})
    for i in range(5):
        assert st.process(observe(bare_head_frame()), now=float(i)) == []
    events = st.process(observe(bare_head_frame()), now=5.0)
    assert events, "should fire once min_track_hits is reached"


def test_compliant_person_never_fires():
    st = state()
    for i in range(20):
        assert st.process(observe(helmeted_frame()), now=float(i)) == []


def test_presence_in_the_window_vetoes():
    """One confident helmet sighting clears the whole window.

    Someone who turns their head and briefly shows a helmet should not be
    reported, even if most frames looked like a bare head.
    """
    st = state()
    for i in range(4):
        st.process(observe(bare_head_frame()), now=float(i))
    st.process(observe(helmeted_frame()), now=4.0)
    events = st.process(observe(bare_head_frame()), now=5.0)
    assert events == [], "a PRESENT frame must veto accumulated evidence"


def test_unknown_frames_do_not_accumulate():
    """A person we can't assess is not evidence of anything."""
    st = state()
    for i in range(12):
        events = st.process(observe([det("person", NEAR_PERSON, 0.91)]), now=float(i))
        assert events == []


def test_absence_only_item_needs_a_stricter_majority():
    """safety-vest at 8-of-10 vs helmet at 5-of-10."""
    st = state()
    first_fire = None
    for i in range(12):
        events = st.process(
            observe([det("person", NEAR_PERSON, 0.91)], required=["safety-vest"]),
            now=float(i),
        )
        if events:
            first_fire = i
            break
    assert first_fire == 7, f"expected the 8th frame, got the {(first_fire or 0) + 1}th"


def test_moving_frames_are_not_recorded():
    """record=False keeps identity but refuses to accumulate evidence.

    This is what makes N-of-M honest: frames taken while the robot drives or
    pans show different scenes and must not count toward agreement.
    """
    st = state()
    for i in range(20):
        events = st.process(observe(bare_head_frame()), now=float(i), record=False)
        assert events == []
    assert len(st.tracker.tracks) == 1, "identity should still be maintained"


# --------------------------------------------------------------------------
# de-duplication
# --------------------------------------------------------------------------


def test_one_person_reports_once_then_cools_down():
    """The anti-spam guarantee: standing in view must not yield 100 uploads."""
    st = state()
    events = []
    for i in range(60):
        events.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert len(events) == 1, f"expected a single report, got {len(events)}"


def test_reports_again_after_the_cooldown_expires():
    """A still-uncompliant person is re-reported once the hold lapses."""
    st = state(cooldown_kw={"per_person_s": 20.0})
    first = []
    for i in range(10):
        first.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert len(first) == 1
    assert st.tracker.tracks[1].item_cooldown["helmet"] == 24.0

    # Fires once more as soon as the cooldown lapses, then holds again. The
    # window stops just short of the NEXT cooldown expiry (24 + 20 = 44 at the
    # earliest), so exactly one report is correct here.
    second = []
    for i in range(30, 44):
        second.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert len(second) == 1, f"expected one repeat report, got {len(second)}"


def test_persisting_violation_reports_once_per_cooldown():
    """Rate of repeat reporting is governed by per_person_s, not frame rate.

    Someone standing in view without a helmet for two minutes should produce a
    handful of periodic reports, not hundreds -- and not exactly one either,
    since an ongoing violation is worth re-surfacing.
    """
    st = state(cooldown_kw={"per_person_s": 30.0})
    events = []
    for i in range(121):  # ~2 minutes at 1 fps
        events.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert 3 <= len(events) <= 5, (
        f"expected roughly one report per 30s cooldown, got {len(events)}"
    )


def test_global_min_interval_throttles_separate_people():
    """Two different people violating at once must not both fire instantly."""
    st = state(cooldown_kw={"global_min_interval_s": 30.0})
    a = (40.0, 100.0, 180.0, 400.0)
    a_head = (85.0, 110.0, 135.0, 170.0)
    emitted = []
    for i in range(10):
        dets = bare_head_frame() + [
            det("person", a, 0.90),
            det("head", a_head, 0.88),
        ]
        emitted.extend(st.process(observe(dets), now=float(i)))
    assert len(emitted) == 1, "global interval should hold the second report back"


def test_max_per_hour_is_a_hard_cap():
    st = state(
        temporal_kw={"positive_n": 1, "min_track_hits": 1},
        cooldown_kw={"per_person_s": 0.1, "global_min_interval_s": 0.0, "max_per_hour": 3},
    )
    emitted = []
    for i in range(40):
        # Reset only the TRACKER, so every frame looks like a brand-new person
        # (worst-case id churn). Resetting the whole state would also clear the
        # rate limiter, which is the thing under test.
        st.tracker.reset()
        emitted.extend(st.process(observe(bare_head_frame()), now=float(i)))
    assert len(emitted) == 3, f"cap is 3/hour, emitted {len(emitted)}"


def test_second_item_on_a_reported_track_still_fires():
    """Cooldown is per item, so a NEW violation is not swallowed."""
    st = state(temporal_kw={"positive_n": 2, "min_track_hits": 2})
    for i in range(6):
        st.process(observe(bare_head_frame(), required=["helmet"]), now=float(i))

    # Same person, now also assessed for gloves with bare hands visible.
    dets = bare_head_frame() + [det("hands", (255.0, 250.0, 300.0, 300.0), 0.80)]
    events = []
    for i in range(6, 14):
        events.extend(
            st.process(observe(dets, required=["helmet", "gloves"]), now=float(i))
        )
    assert any("gloves" in e.items for e in events), "a new item should report"
    assert not any("helmet" in e.items for e in events), "helmet already reported"


# --------------------------------------------------------------------------
# evidence selection
# --------------------------------------------------------------------------


def test_best_frame_is_the_most_confident_one():
    """The snapshot should be the clearest frame, not the one that tipped N."""
    st = state()
    confs = [0.60, 0.95, 0.62, 0.64, 0.66]
    for i, c in enumerate(confs):
        dets = [det("person", NEAR_PERSON, c), det("head", NEAR_HEAD, 0.88)]
        events = st.process(observe(dets), now=float(i))
        if events:
            assert abs(events[0].observation.person.conf - 0.95) < 1e-6, (
                "evidence should come from the highest-confidence frame"
            )
            return
    raise AssertionError("never confirmed")


def test_event_json_round_trips():
    st = state()
    for i in range(10):
        events = st.process(observe(bare_head_frame()), now=float(i))
        if events:
            payload = events[0].to_json()
            assert payload["missing_ppe"] == ["helmet"]
            assert payload["track_id"] == 1
            assert "person_bbox" in payload and len(payload["person_bbox"]) == 4
            assert payload["items"]["helmet"]["status"] == "MISSING_POSITIVE"
            return
    raise AssertionError("never confirmed")


# --------------------------------------------------------------------------
# confirmer unit-level
# --------------------------------------------------------------------------


def test_confirmer_counts_are_reported_accurately():
    tracker = PersonTracker(window=10)
    confirmer = ViolationConfirmer(temporal(), cooldown())
    for i in range(5):
        pairs = tracker.update(observe(bare_head_frame()), now=float(i))
        track = pairs[0][0]
    positive, absent, present, total = track.counts("helmet")
    assert (positive, absent, present, total) == (5, 0, 0, 5)
    assert confirmer.confirmed_items(track) == {"helmet": "5/5"}


def test_window_slides_and_drops_old_verdicts():
    tracker = PersonTracker(window=4)
    for i in range(10):
        pairs = tracker.update(observe(bare_head_frame()), now=float(i))
    track = pairs[0][0]
    assert track.counts("helmet")[3] == 4, "history must be capped at the window"


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = [
        (n, f) for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except BaseException as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    print("PPE tracker / confirmation tests\n")
    sys.exit(_run())
