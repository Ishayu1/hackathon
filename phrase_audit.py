"""Audit common phrases against the fast rule-based classifier."""

from __future__ import annotations

from transcriber import classify_message


PHRASES = [
    # Benign / false-positive-prone
    "i can help lift that box",
    "can you help with homework",
    "please help me carry groceries",
    "get back to me when you can",
    "open fire hydrant inspection is complete",
    "under cover of darkness we moved quietly",
    "the movie was fire",
    "this is a drill",
    "routine training schedule update",
    "the bridge game starts at seven",
    "we need fuel for the generator",
    "hold on, i will call you later",
    # Ambiguous
    "i can help",
    "here i'll go help",
    "get back now",
    "under cover",
    "move to the bridge",
    "the package is at sector seven",
    "raven is moving",
    "checkpoint is delayed",
    # Distress / emergency
    "we need help please help",
    "send help now",
    "mayday mayday taking fire",
    "troops in contact multiple casualties",
    "get out of here now",
    "run away",
    "take cover",
    "under fire need medevac",
    "ied spotted near the convoy",
    "chemical alert evacuate immediately",
    # Command / military ops
    "open fire",
    "cease fire",
    "hold position",
    "secure the objective",
    "withdraw to rally point",
    "execute fire mission",
    "move to grid alpha",
    "defend checkpoint bravo",
    # Intelligence / logistics / admin
    "enemy drone sighting near the bridge",
    "unknown vehicle movement at grid nine",
    "resupply ammo and fuel at checkpoint",
    "medical evacuation requested",
    "authenticate with code word raven",
    "status report from convoy",
]


def main() -> None:
    custom_keywords = "raven, sector seven, bridge"
    for phrase in PHRASES:
        category, severity, score, terms = classify_message(
            phrase,
            custom_keywords=custom_keywords,
        )
        print(
            f"{phrase:<48} -> "
            f"{category.value:<14} {severity.value:<8} {score:>3} {terms}"
        )


if __name__ == "__main__":
    main()
