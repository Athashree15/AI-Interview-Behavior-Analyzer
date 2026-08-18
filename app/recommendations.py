"""
Recommendation generation — every recommendation below is triggered by
an actual measured value from the analyzed video, never a generic or
random suggestion (Module 16 requirement: "based on observed metrics
rather than generic random advice").

Thresholds used here are reasonable, documented defaults (not
empirically tuned against a validation set, since we have no
ground-truth "good interview behavior" labels to tune against) — this
limitation is stated explicitly so it isn't mistaken for a calibrated
clinical/HR threshold.
"""

from __future__ import annotations

# Thresholds are intentionally conservative and documented, not tuned.
EYE_CONTACT_LOW_THRESHOLD = 0.5
HEAD_MOVEMENT_HIGH_THRESHOLD_DEG = 15.0
NEUTRAL_EXPRESSION_HIGH_THRESHOLD = 0.9
HAPPY_EXPRESSION_LOW_THRESHOLD = 0.05


def generate_recommendations(raw_features: dict[str, float]) -> list[str]:
    """
    Generate recommendations strictly from the video's own measured
    feature values. Returns at least one item (a neutral "nothing
    flagged" message if no threshold was crossed) rather than an empty
    list, so the UI always has something concrete to show.
    """
    recommendations = []

    eye_contact = raw_features.get("overall_eye_contact_ratio")
    if eye_contact is not None and eye_contact < EYE_CONTACT_LOW_THRESHOLD:
        recommendations.append(
            f"Approximate eye-contact ratio was {eye_contact:.0%} — consider looking toward the "
            "camera more consistently while responding. (Note: this is an approximate indicator, "
            "not calibrated gaze tracking — see the Responsible AI note below.)"
        )

    yaw_std = raw_features.get("head_yaw_stability_std_deg")
    if yaw_std is not None and yaw_std > HEAD_MOVEMENT_HIGH_THRESHOLD_DEG:
        recommendations.append(
            f"Horizontal head movement was relatively high (std. dev. {yaw_std:.1f}°) — "
            "steadier framing can read as more composed during responses."
        )

    pitch_std = raw_features.get("head_pitch_stability_std_deg")
    if pitch_std is not None and pitch_std > HEAD_MOVEMENT_HIGH_THRESHOLD_DEG:
        recommendations.append(
            f"Vertical head movement was relatively high (std. dev. {pitch_std:.1f}°) — "
            "consider minimizing looking down/nodding excessively while answering."
        )

    neutral = raw_features.get("emotion_neutral")
    happy = raw_features.get("emotion_happy") or 0.0
    if neutral is not None and neutral > NEUTRAL_EXPRESSION_HIGH_THRESHOLD and happy < HAPPY_EXPRESSION_LOW_THRESHOLD:
        recommendations.append(
            f"Facial expression was predominantly neutral ({neutral:.0%} of frames) — "
            "a bit more visible expressiveness where appropriate may read as more engaged."
        )

    if not recommendations:
        recommendations.append(
            "No strong behavioral signals were flagged in the analyzed visual metrics — "
            "the observed patterns fall within the typical ranges used by this tool."
        )

    return recommendations
