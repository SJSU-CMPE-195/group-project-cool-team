"""Person tracking, temporal confirmation and de-duplication.

A single frame is not enough to accuse someone. One bad frame -- motion blur,
an odd pose, a momentary occlusion -- will produce a wrong verdict sooner or
later, and a robot that uploads every one of them fills the portal with noise
until nobody reads it. So a violation has to survive agreement across several
frames of the SAME person before it counts.

That requires three things, which is what this module provides:

  1. Identity. Greedy IoU matching gives each person a stable track id across
     frames. No Kalman filter, no scipy, no filterpy -- a patrol robot sees a
     handful of slow-moving people, and a dependency-free tracker is worth far
     more here than sub-pixel accuracy.

  2. Agreement. Per track, per item, a ring buffer of the last M verdicts.
     A violation fires at N-of-M, with a stricter N for absence-only items.

  3. De-duplication, in three layers, because each defeats a different
     failure:
        per-track cooldown  -- the normal case
        histogram re-ID     -- a track lost and reacquired under a new id,
                               which would otherwise re-report the same person
        global rate limit    -- a backstop for tracker id churn

IMPORTANT: only feed frames captured while the robot is stopped and the camera
has settled. N-of-M over frames taken while panning across a room compares
different scenes and means nothing. ppe_patrol.py enforces this via its
camera_steady / hold_granted handshake.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from ppe_classes import is_positive_evidence_item
from ppe_config import CooldownConfig, TemporalConfig
from ppe_rules import ItemStatus, PersonObservation
from ppe_types import Box, iou

# Set by ppe_patrol when a real frame is available; kept optional so the whole
# module is testable with no numpy and no images.
try:  # pragma: no cover
    import cv2
    import numpy as np

    _HAVE_CV = True
except ImportError:  # pragma: no cover
    _HAVE_CV = False


@dataclass
class ViolationEvent:
    """A confirmed violation, ready to be snapshotted and uploaded."""

    track_id: int
    items: list[str]
    observation: PersonObservation
    frames_agree: dict[str, str]  # item -> "6/10"
    confirmed_at: float

    def to_json(self) -> dict:
        return {
            "track_id": self.track_id,
            "missing_ppe": list(self.items),
            "frames_agree": dict(self.frames_agree),
            **self.observation.to_json(),
        }


@dataclass
class Track:
    """One person followed across frames."""

    track_id: int
    box: Box
    # 0, not 1: update() increments every matched track including one it just
    # created, so a fresh track would otherwise be credited with two frames
    # after its first sighting and clear min_track_hits a frame early.
    hits: int = 0
    misses: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    history: dict[str, deque[ItemStatus]] = field(default_factory=dict)
    # Highest-confidence observation seen while this track was confirming, so
    # the evidence image is the clearest frame rather than whichever one
    # happened to tip the counter over.
    best_obs: PersonObservation | None = None
    best_frame: object | None = None
    best_score: float = -1.0
    # Cooldown is per ITEM, not per track. A track-wide hold would swallow a
    # genuinely new violation -- someone already reported for no helmet, whom
    # we then notice also has no gloves.
    item_cooldown: dict[str, float] = field(default_factory=dict)
    signature: object | None = None

    def suppressed(self, item: str, now: float) -> bool:
        return now < self.item_cooldown.get(item, float("-inf"))

    def record(self, item: str, status: ItemStatus, window: int) -> None:
        buf = self.history.get(item)
        if buf is None or buf.maxlen != window:
            buf = deque(buf or (), maxlen=window)
            self.history[item] = buf
        buf.append(status)

    def counts(self, item: str) -> tuple[int, int, int, int]:
        """(positive, absent, present, total) over the window for `item`."""
        buf = self.history.get(item, ())
        positive = sum(1 for s in buf if s == "MISSING_POSITIVE")
        absent = sum(1 for s in buf if s == "MISSING_ABSENT")
        present = sum(1 for s in buf if s == "PRESENT")
        return positive, absent, present, len(buf)

    def clear_item(self, item: str) -> None:
        self.history.pop(item, None)


# --------------------------------------------------------------------------
# appearance signature, for re-identification
# --------------------------------------------------------------------------


def hist_signature(frame_bgr, box: Box):
    """Coarse colour signature of a person crop, for naive re-ID.

    A 2x2 grid of 16-bin hue histograms, L2-normalised. Enough to recognise
    "same person, brief occlusion" within a couple of minutes, which is the
    case that would otherwise re-report someone under a fresh track id. It
    will NOT survive a change of clothes or a big lighting shift, and it is
    not meant to -- it exists to suppress duplicates, never to identify
    anyone.
    """
    if not _HAVE_CV or frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ch, cw = hsv.shape[:2]
    cells = []
    for gy in range(2):
        for gx in range(2):
            cell = hsv[
                gy * ch // 2 : (gy + 1) * ch // 2,
                gx * cw // 2 : (gx + 1) * cw // 2,
            ]
            if cell.size == 0:
                cells.append(np.zeros(16, dtype=np.float32))
                continue
            # Mask out near-black and washed-out pixels: their hue is noise.
            mask = cv2.inRange(cell, (0, 40, 40), (180, 255, 255))
            hist = cv2.calcHist([cell], [0], mask, [16], [0, 180]).flatten()
            cells.append(hist.astype(np.float32))
    sig = np.concatenate(cells)
    norm = float(np.linalg.norm(sig))
    return sig / norm if norm > 1e-6 else None


def hist_similarity(a, b) -> float:
    """Cosine similarity of two signatures; 0.0 if either is missing."""
    if a is None or b is None or not _HAVE_CV:
        return 0.0
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


# --------------------------------------------------------------------------
# tracker
# --------------------------------------------------------------------------


class PersonTracker:
    """Greedy IoU tracker with a centroid fallback for fast motion."""

    def __init__(
        self,
        iou_thresh: float = 0.30,
        max_misses: int = 15,
        window: int = 10,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.window = window
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(
        self,
        observations: list[PersonObservation],
        now: float,
        frame_bgr=None,
        record: bool = True,
    ) -> list[tuple[Track, PersonObservation]]:
        """Match observations to tracks, ageing out the unmatched.

        `record=False` still maintains identity but does not feed the
        temporal history -- used while the robot is moving, so tracks survive
        the journey without accumulating meaningless verdicts.
        """
        pairs = self._match(observations)
        matched_ids: set[int] = set()

        out: list[tuple[Track, PersonObservation]] = []
        for track, obs in pairs:
            track.box = obs.person.xyxy
            track.hits += 1
            track.misses = 0
            track.last_seen = now
            matched_ids.add(track.track_id)

            if record:
                for item, evidence in obs.items.items():
                    track.record(item, evidence.status, self.window)
                self._maybe_keep_best(track, obs, frame_bgr)
            if track.signature is None and frame_bgr is not None:
                track.signature = hist_signature(frame_bgr, track.box)
            out.append((track, obs))

        for track_id, track in list(self.tracks.items()):
            if track_id in matched_ids:
                continue
            track.misses += 1
            if track.misses > self.max_misses:
                del self.tracks[track_id]
        return out

    def _match(
        self, observations: list[PersonObservation]
    ) -> list[tuple[Track, PersonObservation]]:
        if not observations:
            return []

        candidates: list[tuple[float, int, int]] = []
        obs_list = list(observations)
        for oi, obs in enumerate(obs_list):
            for track_id, track in self.tracks.items():
                score = iou(obs.person.xyxy, track.box)
                if score < self.iou_thresh:
                    score = self._centroid_score(obs.person.xyxy, track.box)
                if score > 0.0:
                    candidates.append((score, oi, track_id))

        candidates.sort(reverse=True)
        used_obs: set[int] = set()
        used_tracks: set[int] = set()
        pairs: list[tuple[Track, PersonObservation]] = []
        for _, oi, track_id in candidates:
            if oi in used_obs or track_id in used_tracks:
                continue
            used_obs.add(oi)
            used_tracks.add(track_id)
            pairs.append((self.tracks[track_id], obs_list[oi]))

        for oi, obs in enumerate(obs_list):
            if oi in used_obs:
                continue
            track = Track(track_id=self._next_id, box=obs.person.xyxy)
            self._next_id += 1
            self.tracks[track.track_id] = track
            pairs.append((track, obs))
        return pairs

    def _centroid_score(self, a: Box, b: Box) -> float:
        """Fallback for a person who moved too far for IoU to match.

        Accepts a match within half the track box's diagonal, scoring below
        any real IoU match so genuine overlaps always win.
        """
        ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
        bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        diag = ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
        if diag <= 0 or dist > 0.5 * diag:
            return 0.0
        return 0.01 * (1.0 - dist / (0.5 * diag))

    def _maybe_keep_best(self, track: Track, obs: PersonObservation, frame_bgr) -> None:
        """Cache the clearest confirming frame as the evidence image."""
        missing = obs.missing_items()
        if not missing:
            return
        score = obs.person.conf + max(
            (max(ev.support.values(), default=0.0) for ev in obs.items.values()),
            default=0.0,
        )
        if score > track.best_score:
            track.best_score = score
            track.best_obs = obs
            if frame_bgr is not None:
                track.best_frame = frame_bgr.copy() if _HAVE_CV else frame_bgr

    def reset(self) -> None:
        self.tracks.clear()


# --------------------------------------------------------------------------
# confirmation and de-duplication
# --------------------------------------------------------------------------


class ViolationConfirmer:
    """Decides when agreement is enough, and suppresses duplicate reports."""

    def __init__(
        self,
        temporal: TemporalConfig,
        cooldown: CooldownConfig,
    ) -> None:
        self.temporal = temporal
        self.cooldown = cooldown
        # -inf, not 0.0: a fresh confirmer has never emitted, and 0.0 would
        # make the global interval suppress everything for the first N seconds
        # of a run (and forever, for a monotonic clock starting near zero).
        self._last_emit = float("-inf")
        self._emit_times: deque[float] = deque()
        # (signature, items, expires_at) for people already reported, so a
        # reacquired track inherits their cooldown instead of re-reporting.
        self._recent: list[tuple[object, set[str], float]] = []

    # ------------------------------------------------------------------ gate

    def confirmed_items(self, track: Track) -> dict[str, str]:
        """Items meeting the N-of-M bar. Returns item -> "agree/total"."""
        out: dict[str, str] = {}
        if track.hits < self.temporal.min_track_hits:
            return out

        for item in list(track.history):
            positive, absent, present, total = track.counts(item)
            if total == 0 or present > 0:
                # Presence veto across the window: one confident sighting of
                # the PPE clears the item outright.
                continue
            if is_positive_evidence_item(item):
                if positive >= self.temporal.positive_n:
                    out[item] = f"{positive}/{total}"
            else:
                # Absence-only: weaker evidence, stricter bar, and positive
                # frames count toward it too.
                agree = positive + absent
                if agree >= self.temporal.absence_n:
                    out[item] = f"{agree}/{total}"
        return out

    def feed(self, track: Track, now: float) -> ViolationEvent | None:
        """Emit an event if this track has newly confirmed a violation."""
        confirmed = self.confirmed_items(track)
        if not confirmed:
            return None

        fresh = {
            item: agree
            for item, agree in confirmed.items()
            if not track.suppressed(item, now)
        }
        if not fresh:
            return None
        if not self._rate_ok(now):
            return None

        obs = track.best_obs
        if obs is None:
            return None

        items = sorted(fresh)
        self._note_emit(track, items, now)
        return ViolationEvent(
            track_id=track.track_id,
            items=items,
            observation=obs,
            frames_agree=fresh,
            confirmed_at=now,
        )

    # --------------------------------------------------------------- helpers

    def _rate_ok(self, now: float) -> bool:
        if now - self._last_emit < self.cooldown.global_min_interval_s:
            return False
        while self._emit_times and now - self._emit_times[0] > 3600.0:
            self._emit_times.popleft()
        return len(self._emit_times) < self.cooldown.max_per_hour

    def _note_emit(self, track: Track, items: list[str], now: float) -> None:
        for item in items:
            track.item_cooldown[item] = now + self.cooldown.per_person_s
            # Clear the window so re-accumulation starts clean after cooldown.
            track.clear_item(item)
        self._last_emit = now
        self._emit_times.append(now)
        if track.signature is not None:
            self._recent.append(
                (track.signature, set(items), now + self.cooldown.reid_window_s)
            )
        self._prune_recent(now)

    def _prune_recent(self, now: float) -> None:
        self._recent = [r for r in self._recent if r[2] > now]

    def adopt_history(self, track: Track, now: float) -> bool:
        """Give a newly-born track the cooldown of a person we just reported.

        Called on track birth. Without this, someone who walks behind a chair
        and reappears gets a fresh id and is reported again immediately.
        """
        if track.signature is None:
            return False
        self._prune_recent(now)
        for signature, items, expires in self._recent:
            if hist_similarity(track.signature, signature) >= self.cooldown.reid_similarity:
                for item in items:
                    track.item_cooldown[item] = max(
                        track.item_cooldown.get(item, float("-inf")), expires
                    )
                return True
        return False

    def reset(self) -> None:
        self._recent.clear()
        self._emit_times.clear()
        self._last_emit = float("-inf")


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------


class PatrolVisionState:
    """Tracker + confirmer, wired together.

    Convenience wrapper so ppe_patrol can hand over a frame's observations and
    receive back any confirmed violations.
    """

    def __init__(self, temporal: TemporalConfig, cooldown: CooldownConfig) -> None:
        self.tracker = PersonTracker(window=temporal.window_m)
        self.confirmer = ViolationConfirmer(temporal, cooldown)
        self._known: set[int] = set()

    def process(
        self,
        observations: list[PersonObservation],
        now: float,
        frame_bgr=None,
        record: bool = True,
    ) -> list[ViolationEvent]:
        pairs = self.tracker.update(observations, now, frame_bgr, record=record)
        events: list[ViolationEvent] = []
        for track, _obs in pairs:
            if track.track_id not in self._known:
                self._known.add(track.track_id)
                self.confirmer.adopt_history(track, now)
            if not record:
                continue
            event = self.confirmer.feed(track, now)
            if event is not None:
                events.append(event)
        self._known &= set(self.tracker.tracks)
        return events

    def reset(self) -> None:
        self.tracker.reset()
        self.confirmer.reset()
        self._known.clear()

    def describe(self) -> str:
        bits: list[str] = []
        for track in self.tracker.tracks.values():
            per_item = ",".join(
                f"{item}:{'/'.join(map(str, track.counts(item)[:2]))}"
                for item in track.history
            )
            bits.append(f"t{track.track_id}(h{track.hits} {per_item or '-'})")
        return " ".join(bits) or "no tracks"


def iter_tracks(tracker: PersonTracker) -> Iterable[Track]:
    return tracker.tracks.values()
