"""A small, deterministic heuristic for the Triad UNO starter agent.

The strategy only chooses among the request's legal actions.  It uses public
hand counts to pressure the next player, sheds high-value cards, and keeps
wild cards when a similarly useful ordinary play is available.
"""

from __future__ import annotations

from typing import Any


COLORS = ("red", "yellow", "green", "blue")
ACTION_POINTS = {"skip": 20, "reverse": 20, "draw_two": 20,
                 "wild": 50, "wild_draw_four": 50}

# Exposed so local_evaluator training scripts can tune the policy without
# changing its decision code. Freeze the selected values before submission.
PARAMS: dict[str, float] = {
    "card_points": 0.07667973150050385,
    "hand_points": 0.30231071994945224,
    "last_card": 4.487723406887613,
    "color_count": 2.902461727001265,
    "color_concentration": 0.18340658694372203,
    "opponent_color_shortage": 1.2193334973585266,
    "matching_card_support": 1.0612640707177126,
    "wild_penalty": 24.535550389352352,
    "wild_draw_four_penalty": 25.3163657692644,
    "action_penalty": 1.0603276978304235,
    "threat_one": 10.530703452696068,
    "threat_two": 9.087621646667504,
    "wild_draw_four_threat": 6.05809640837192,
    "endgame_points": 0.20690289017681518,
}


def _card_for(action: dict[str, Any]) -> dict[str, Any]:
    card = action.get("card")
    return card if isinstance(card, dict) else {}


def _card_kind(action: dict[str, Any]) -> str:
    return str(_card_for(action).get("kind", ""))


def _card_color(action: dict[str, Any]) -> str | None:
    color = _card_for(action).get("color")
    return color if color in COLORS else None


def _card_points(action: dict[str, Any]) -> int:
    card = _card_for(action)
    kind = card.get("kind")
    if kind == "number":
        value = card.get("value")
        return value if isinstance(value, int) else 0
    return ACTION_POINTS.get(str(kind), 0)


def _next_unfinished(seat: int, direction: int,
                     hand_sizes: list[Any], finished: set[int]) -> int | None:
    """Return the next seat that can act, following engine seat order."""
    for distance in range(1, len(hand_sizes) + 1):
        candidate = (seat + direction * distance) % len(hand_sizes)
        if candidate not in finished and candidate != seat:
            return candidate
    return None


def _target_for(action: dict[str, Any], observation: dict[str, Any]) -> int | None:
    seat = observation.get("seat")
    if not isinstance(seat, int):
        return None
    sizes = observation.get("hand_sizes")
    if not isinstance(sizes, list) or not sizes:
        return None
    finished_raw = observation.get("finished_order") or []
    finished = {value for value in finished_raw if isinstance(value, int)}
    direction = observation.get("direction", 1)
    if not isinstance(direction, int) or direction not in (-1, 1):
        direction = 1
    if _card_kind(action) == "reverse":
        direction *= -1
    return _next_unfinished(seat, direction, sizes, finished)


def _color_counts(hand: list[dict[str, Any]], excluded_ref: Any = None) -> dict[str, int]:
    counts = {color: 0 for color in COLORS}
    for card in hand:
        if card.get("card_ref") == excluded_ref:
            continue
        color = card.get("color")
        if color in counts:
            counts[color] += 1
    return counts


def _next_player(observation: dict[str, Any], direction: int | None = None) -> int | None:
    """Find the next active seat using only public turn order and finish data."""
    seat = observation.get("seat")
    sizes = observation.get("hand_sizes")
    if not isinstance(seat, int) or not isinstance(sizes, list) or not sizes:
        return None
    if direction is None:
        direction = observation.get("direction", 1)
    if not isinstance(direction, int) or direction not in (-1, 1):
        direction = 1
    finished_raw = observation.get("finished_order") or []
    finished = {value for value in finished_raw if isinstance(value, int)}
    return _next_unfinished(seat, direction, sizes, finished)


def _color_shortage_evidence(observation: dict[str, Any],
                             target: int | None) -> dict[str, float]:
    """Estimate colors a rival may lack from public failed-draw history.

    A normal draw is offered only when a player has no legal play. Since a
    Wild is always playable, this is evidence they lacked the active color.
    Subsequent plays and draws weaken old evidence. This remains probabilistic:
    newly drawn or previously unseen cards can change a rival's hand.
    """
    evidence = {color: 0.0 for color in COLORS}
    if target is None:
        return evidence

    active_color: str | None = None
    for event in observation.get("history") or []:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")

        if event_type == "game_start":
            initial = event.get("initial_discard") or {}
            active_color = initial.get("color") if isinstance(initial, dict) else None
        elif event_type == "initial_discard":
            initial = event.get("card") or {}
            active_color = initial.get("color") if isinstance(initial, dict) else None
        elif event_type == "initial_color_choice":
            active_color = event.get("color")

        event_seat = event.get("seat")
        if event_type == "draw" and event_seat == target:
            if active_color in evidence:
                evidence[active_color] = min(2.0, evidence[active_color] + 1.0)
        elif event_type == "penalty_draw" and event_seat == target:
            cards = event.get("count", event.get("penalty", 1))
            cards = cards if isinstance(cards, int) and cards > 0 else 1
            decay = 0.55 ** min(cards, 4)
            evidence = {color: value * decay for color, value in evidence.items()}
        elif event_type == "initial_draw_two" and event.get("penalized") == target:
            evidence = {color: value * 0.55 ** 2 for color, value in evidence.items()}
        elif event_type == "play" and event_seat == target:
            # A play immediately after a draw is the newly drawn card; it does
            # not invalidate the evidence that their previous hand lacked color.
            if not event.get("from_draw"):
                evidence = {color: value * 0.65 for color, value in evidence.items()}
        elif event_type == "keep" and event_seat == target:
            # They retained a playable drawn card, which may have restored a
            # color that was missing when they elected to draw.
            evidence = {color: value * 0.65 for color, value in evidence.items()}

        if event_type == "play":
            played = event.get("active_color")
            if played in evidence:
                active_color = played

    return evidence


def _choose_color(hand: list[dict[str, Any]],
                  observation: dict[str, Any] | None = None) -> str:
    """Favor colors in hand, with a small adjustment for public rival evidence."""
    counts = _color_counts(hand)
    if observation is None:
        return max(COLORS, key=lambda color: counts[color])
    shortages = _color_shortage_evidence(observation, _next_player(observation))
    return max(
        COLORS,
        key=lambda color: (
            counts[color] * PARAMS["color_count"]
            + shortages[color] * PARAMS["opponent_color_shortage"]
        ),
    )


def _pending_penalty(observation: dict[str, Any]) -> int:
    """Read the current penalty, reconstructing it when not explicitly sent."""
    reported = observation.get("pending_penalty")
    if isinstance(reported, int) and reported >= 0:
        return reported
    total = 0
    for event in observation.get("history") or []:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "play":
            kind = (event.get("card") or {}).get("kind")
            if kind == "draw_two":
                total += 2
            elif kind == "wild_draw_four":
                total += 4
        elif event.get("type") == "penalty_draw":
            total = 0
    return total


def _score_play(action: dict[str, Any], observation: dict[str, Any],
                hand: list[dict[str, Any]], hand_sizes: list[Any],
                shortage_evidence: dict[str, float] | None = None) -> float:
    """Score a legal play; larger scores are preferred."""
    card = _card_for(action)
    kind = str(card.get("kind", ""))
    ref = action.get("card_ref")
    cards_left = len(hand) - 1
    if cards_left <= 0:
        return 10000.0

    # Points only break close choices: getting closer to emptying the hand is
    # more useful than racing to shed a high-value card too early.
    score = _card_points(action) * (
        PARAMS["card_points"] + PARAMS["hand_points"] / (len(hand) + 1)
    )
    if cards_left == 1:
        score += PARAMS["last_card"]

    chosen_color = action.get("color")
    if chosen_color not in COLORS:
        chosen_color = _card_color(action)
    counts = _color_counts(hand, ref)
    if chosen_color in COLORS:
        score += counts[chosen_color] * PARAMS["color_count"]
        # Prefer keeping a color that remains common after this play.
        score += (
            counts[chosen_color]
            * PARAMS["color_concentration"]
            * max(counts.values(), default=0)
        )
        # Retaining a same-rank/action card gives another way to answer if the
        # next discard changes color before our following turn.
        matching_support = sum(
            1 for held in hand
            if held.get("card_ref") != ref
            and (
                (kind == "number" and held.get("kind") == "number"
                 and held.get("value") == card.get("value"))
                or (kind in {"skip", "reverse", "draw_two"}
                    and held.get("kind") == kind)
            )
        )
        score += matching_support * PARAMS["matching_card_support"]
        if kind in {"wild", "wild_draw_four"}:
            # A color an opponent recently failed to play is a useful Wild
            # choice, especially when that opponent is close to finishing.
            target = _target_for(action, observation)
            shortages = shortage_evidence or _color_shortage_evidence(observation, target)
            sizes = observation.get("hand_sizes")
            target_size = (
                sizes[target]
                if target is not None and isinstance(sizes, list)
                and target < len(sizes) else None
            )
            multiplier = 1.5 if target_size == 1 else 1.0
            score += (
                shortages.get(chosen_color, 0.0)
                * PARAMS["opponent_color_shortage"]
                * multiplier
            )

    # Wilds are valuable outs. Spend one when it improves the color position,
    # prevents an imminent finish, or is the only available legal play.
    if kind == "wild":
        score -= PARAMS["wild_penalty"]
    elif kind == "wild_draw_four":
        score -= PARAMS["wild_draw_four_penalty"]
    elif kind in {"skip", "reverse", "draw_two"}:
        score -= PARAMS["action_penalty"]

    target = _target_for(action, observation)
    if target is not None and 0 <= target < len(hand_sizes):
        target_count = hand_sizes[target]
        if isinstance(target_count, int) and target_count <= 2:
            urgency = (
                PARAMS["threat_one"] if target_count == 1 else PARAMS["threat_two"]
            )
            if kind in {"skip", "reverse", "draw_two", "wild_draw_four"}:
                score += urgency
            if kind == "wild_draw_four":
                score += PARAMS["wild_draw_four_threat"]

    # On the final few turns, emptying the hand soon is more important than
    # retaining a perfectly balanced color distribution.
    if len(hand) <= 3:
        score += (3 - len(hand)) * _card_points(action) * PARAMS["endgame_points"]
    return score


def select_action(observation: dict[str, Any],
                  agent_state: Any) -> tuple[str, Any]:
    """Return a current legal action id and leave the protocol checkpoint intact."""
    legal = observation.get("legal_actions") or []
    if not legal:
        # The protocol always supplies a legal action; keep this defensive
        # branch harmless for direct callers with incomplete observations.
        return "a0", agent_state

    phase = observation.get("phase")
    hand = observation.get("hand")
    if not isinstance(hand, list):
        hand = []

    if phase == "CHOOSE_INITIAL_COLOR":
        preferred = _choose_color(hand, observation)
        for action in legal:
            if action.get("type") == "choose_color" and action.get("color") == preferred:
                return str(action["action_id"]), agent_state
        return str(legal[0]["action_id"]), agent_state

    if phase == "AFTER_DRAW":
        # Keeping a newly drawn Wild Draw Four can be better than spending it
        # immediately when the hand is still large and no opponent is close to
        # finishing. Retain the card as a future guaranteed play. The engine
        # explicitly allows keeping a playable drawn card in this phase.
        drawn_ref = observation.get("drawn_card_ref")
        drawn_card = next((card for card in hand
                           if card.get("card_ref") == drawn_ref), None)
        drawn_wdf_is_playable = any(
            action.get("type") == "play" and action.get("card_ref") == drawn_ref
            and _card_kind(action) == "wild_draw_four"
            for action in legal
        )
        if (drawn_card is not None
                and drawn_card.get("kind") == "wild_draw_four"
                and drawn_wdf_is_playable
                and len(hand) >= 5):
            sizes = observation.get("hand_sizes")
            if not isinstance(sizes, list):
                sizes = []
            seat = observation.get("seat")
            finished = {value for value in (observation.get("finished_order") or [])
                        if isinstance(value, int)}
            opponent_threat = any(
                index != seat and index not in finished
                and isinstance(size, int) and size <= 2
                for index, size in enumerate(sizes)
            )
            keep = next((action for action in legal
                         if action.get("type") == "keep"), None)
            if not opponent_threat and keep is not None:
                return str(keep["action_id"]), agent_state

        plays = [action for action in legal if action.get("type") == "play"]
        if plays:
            # Only the newly drawn card is playable in this phase. Score color
            # variants consistently and retain the existing checkpoint.
            sizes = observation.get("hand_sizes")
            if not isinstance(sizes, list):
                sizes = []
            shortage = _color_shortage_evidence(observation, _next_player(observation))
            best = max(plays, key=lambda action: _score_play(
                action, observation, hand, sizes, shortage
            ))
            return str(best["action_id"]), agent_state
        return str(legal[0]["action_id"]), agent_state

    plays = [action for action in legal if action.get("type") == "play"]
    if not plays:
        return str(legal[0]["action_id"]), agent_state

    # A pending draw chain gives the next player cards and costs no turn. Keep
    # stacking when possible, but use the least flexible multiplier first.
    if phase == "TURN" and _pending_penalty(observation):
        stackable = [action for action in plays
                     if _card_kind(action) in {"draw_two", "wild_draw_four"}]
        if stackable:
            non_wild_stack = [action for action in stackable if _card_kind(action) == "draw_two"]
            choices = non_wild_stack or stackable
            sizes = observation.get("hand_sizes")
            if not isinstance(sizes, list):
                sizes = []
            shortage = _color_shortage_evidence(observation, _next_player(observation))
            best = max(choices, key=lambda action: _score_play(
                action, observation, hand, sizes, shortage
            ))
            return str(best["action_id"]), agent_state

    sizes = observation.get("hand_sizes")
    if not isinstance(sizes, list):
        sizes = []
    shortage = _color_shortage_evidence(observation, _next_player(observation))
    best = max(plays, key=lambda action: _score_play(
        action, observation, hand, sizes, shortage
    ))
    return str(best["action_id"]), agent_state
