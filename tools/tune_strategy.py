#!/usr/bin/env python3
"""Tune the editable UNO heuristic against weak and baseline opponents.

This is a development-only tool. It runs the real in-process evaluator and
prints candidate parameters; copy the selected values into ``src/strategy.py``
before building a submission. Training seeds and validation seeds are disjoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_evaluator" / "src"))

from triad_uno import engine, scoring, simulator  # noqa: E402
from triad_uno.baselines import HeuristicAgent, RandomAgent  # noqa: E402

OPPONENTS = ("heuristic", "random", "first_legal", "projection", "attack_greedy")
KIND_RANK = {
    "number": 0, "skip": 1, "reverse": 2, "draw_two": 3,
    "wild": 4, "wild_draw_four": 5, "draw": 6, "keep": 7,
    "choose_color": 8,
}
COLOR_RANK = {color: index for index, color in enumerate(("red", "yellow", "green", "blue"))}


def load_strategy() -> Any:
    """Load the strategy file directly, without importing submission code."""
    path = ROOT / "src" / "strategy.py"
    spec = importlib.util.spec_from_file_location("tuned_uno_strategy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load strategy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CandidateAgent:
    """Adapt the submission hook to the in-process evaluator interface."""

    def __init__(self, strategy: Any) -> None:
        self.strategy = strategy

    def select(self, observation: dict[str, Any]) -> str:
        """Choose an action using the currently active parameter set."""
        action_id, _ = self.strategy.select_action(observation, None)
        return action_id


class FirstLegalAgent:
    """Simple policy common in starter/vibe-coded agents: always choose a0."""

    def select(self, observation: dict[str, Any]) -> str:
        legal = observation.get("legal_actions") or []
        return str(legal[0]["action_id"]) if legal else "a0"


class ProjectionAgent:
    """Match the bundled weak projection-scoring example without torch."""

    def __init__(self) -> None:
        rng = random.Random(1234)
        self.projection = [
            [rng.uniform(-1.0, 1.0) for _ in range(3)] for _ in range(4)
        ]

    @staticmethod
    def encode(action: dict[str, Any]) -> list[float]:
        card = action.get("card") or {}
        kind = card.get("kind", action.get("type"))
        color = action.get("color") or card.get("color")
        value = card.get("value")
        return [
            float(KIND_RANK.get(str(kind), 0)),
            float(COLOR_RANK.get(str(color), 0)),
            float(value) if isinstance(value, int) else 0.0,
        ]

    def select(self, observation: dict[str, Any]) -> str:
        legal = observation.get("legal_actions") or []
        if not legal:
            return "a0"
        def score(action: dict[str, Any]) -> float:
            values = self.encode(action)
            return sum(max(0.0, sum(x * w for x, w in zip(values, row)))
                       for row in self.projection)
        best = max(legal, key=score)
        return str(best["action_id"])


class AttackGreedyAgent:
    """A weak high-card/attack-first policy often seen in quick bot drafts."""

    def select(self, observation: dict[str, Any]) -> str:
        legal = observation.get("legal_actions") or []
        if not legal:
            return "a0"
        hand = observation.get("hand") or []
        counts = {color: 0 for color in COLOR_RANK}
        for card in hand:
            if card.get("color") in counts:
                counts[card["color"]] += 1
        if observation.get("phase") == "CHOOSE_INITIAL_COLOR":
            color = max(counts, key=counts.get)
            return str(next((a["action_id"] for a in legal if a.get("color") == color),
                            legal[0]["action_id"]))
        plays = [a for a in legal if a.get("type") == "play"]
        if not plays:
            return str(legal[0]["action_id"])
        def score(action: dict[str, Any]) -> tuple[int, int]:
            card = action.get("card") or {}
            kind = card.get("kind")
            points = card.get("value", 0) if kind == "number" else {
                "skip": 20, "reverse": 20, "draw_two": 20,
                "wild": 50, "wild_draw_four": 50,
            }.get(kind, 0)
            attack = {"draw_two": 3, "wild_draw_four": 4,
                      "skip": 2, "reverse": 1}.get(kind, 0)
            return points + attack * 6, counts.get(action.get("color") or card.get("color"), 0)
        return str(max(plays, key=score)["action_id"])


def make_opponent(kind: str, seed: int) -> Any:
    if kind == "random":
        return RandomAgent(seed)
    if kind == "first_legal":
        return FirstLegalAgent()
    if kind == "projection":
        return ProjectionAgent()
    if kind == "attack_greedy":
        return AttackGreedyAgent()
    return HeuristicAgent(seed)


def sample_parameters(rng: random.Random) -> dict[str, float]:
    """Sample practical weights around the current public-information policy."""
    return {
        "card_points": rng.uniform(0.05, 0.25),
        "hand_points": rng.uniform(0.05, 0.50),
        "last_card": rng.uniform(1.5, 6.0),
        "color_count": rng.uniform(2.0, 4.2),
        "color_concentration": rng.uniform(0.05, 0.50),
        "opponent_color_shortage": rng.uniform(0.0, 7.0),
        "matching_card_support": rng.uniform(0.0, 1.5),
        "wild_penalty": rng.uniform(15.0, 25.0),
        "wild_draw_four_penalty": rng.uniform(19.0, 31.0),
        "action_penalty": rng.uniform(0.0, 2.0),
        "threat_one": rng.uniform(10.0, 20.0),
        "threat_two": rng.uniform(4.0, 11.0),
        "wild_draw_four_threat": rng.uniform(2.0, 7.0),
        "endgame_points": rng.uniform(0.05, 0.30),
    }


def play_batch(strategy: Any, parameters: dict[str, float] | None,
               games: int, first_seed: int,
               profile: str = "mixed") -> dict[str, float]:
    """Evaluate with candidate seats rotated evenly against the chosen profile."""
    candidate = CandidateAgent(strategy) if parameters is not None else None
    if parameters is not None:
        strategy.PARAMS.update(parameters)
    total_points = 0.0
    wins = 0
    wild_plays = 0
    exposure = {name: {"candidate_points": 0.0, "games": 0}
                 for name in OPPONENTS}
    for index in range(games):
        # Evaluate every candidate in all four seats on each exact game seed.
        # This pairs the random deal and avoids correlating seat with seed.
        seed = first_seed + index // engine.SEATS
        candidate_seat = index % engine.SEATS
        agents = []
        seat_kinds = []
        for seat in range(engine.SEATS):
            if seat == candidate_seat:
                agents.append(candidate if candidate is not None
                              else HeuristicAgent(seed * 101 + seat))
                seat_kinds.append(None)
                continue
            if profile == "mixed":
                kind = OPPONENTS[(index + seat + first_seed) % len(OPPONENTS)]
            else:
                kind = profile
            seat_kinds.append(kind)
            agents.append(make_opponent(kind, seed * 101 + seat))
        final, _, events = simulator.run_game(seed, agents, engine.GameConfig())
        outcome = final.outcome
        if outcome is None:
            continue
        wild_plays += sum(
            event.get("seat") == candidate_seat
            and event.get("type") == "play"
            and (event.get("card") or {}).get("kind") in {"wild", "wild_draw_four"}
            for event in events
        )
        outcome_dict = {
            "placements": outcome.placements,
            "unfinished": outcome.unfinished,
        }
        seat_names = ["candidate" if seat == candidate_seat else f"opponent_{seat}"
                      for seat in range(engine.SEATS)]
        points = scoring.points_for_game(outcome_dict, seat_names)
        total_points += points["candidate"]
        wins += bool(outcome.placements and outcome.placements[0] == candidate_seat)
        if candidate is not None:
            for seat, kind in enumerate(seat_kinds):
                if kind is not None:
                    exposure[kind]["games"] += 1
                    exposure[kind]["candidate_points"] += points["candidate"]
    result = {
        "average_points": total_points / games,
        "first_places": float(wins),
        "wild_plays_per_game": wild_plays / games,
    }
    if candidate is not None:
        result["points_when_facing"] = {
            name: round(values["candidate_points"] / values["games"], 4)
            if values["games"] else 0.0
            for name, values in exposure.items()
        }
    return result


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save progress so a long training run can resume safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    """Run a seeded random search, then validate the strongest candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--games", type=int, default=800,
                        help="training games per parameter set; use a multiple of four")
    parser.add_argument("--validation-games", type=int, default=1600,
                        help="fresh games for each finalist; use a multiple of four")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--search-seed", type=int, default=20261014)
    parser.add_argument("--profile", choices=("mixed", *OPPONENTS),
                        default="mixed", help="opponent archetype to optimize against")
    parser.add_argument("--wild-risk-scale", type=float, default=1.0,
                        help="scale Wild and Wild Draw Four penalties; below 1 is riskier")
    parser.add_argument("--params-file", type=Path,
                        help="evaluate these weights instead of searching (use --candidates 1)")
    parser.add_argument("--checkpoint-file", type=Path,
                        help="save each completed candidate and resume matching runs")
    args = parser.parse_args()
    if min(args.candidates, args.games, args.validation_games, args.top) < 1:
        parser.error("candidate, game, and finalist counts must be positive")
    if args.games % 4 or args.validation_games % 4:
        parser.error("games and validation-games must be multiples of four for seat rotation")
    if args.params_file is not None and args.candidates != 1:
        parser.error("--params-file requires --candidates 1")

    config = {
        "candidates": args.candidates,
        "games": args.games,
        "validation_games": args.validation_games,
        "top": args.top,
        "seed": args.seed,
        "search_seed": args.search_seed,
        "profile": args.profile,
        "wild_risk_scale": args.wild_risk_scale,
        "params_file": str(args.params_file) if args.params_file else None,
    }
    checkpoint: dict[str, Any] = {
        "config": config,
        "results": [],
        "reference": None,
        "validated": [],
    }
    if args.checkpoint_file is not None and args.checkpoint_file.exists():
        checkpoint = json.loads(args.checkpoint_file.read_text(encoding="utf-8"))
        if checkpoint.get("config") != config:
            parser.error("checkpoint configuration does not match this run")
    elif args.checkpoint_file is not None:
        save_checkpoint(args.checkpoint_file, checkpoint)

    strategy = load_strategy()
    original_params = dict(strategy.PARAMS)
    rng = random.Random(args.search_seed)
    if args.params_file is not None:
        candidates = [json.loads(args.params_file.read_text(encoding="utf-8"))]
    else:
        candidates = [dict(original_params)]
        candidates[0]["wild_penalty"] *= args.wild_risk_scale
        candidates[0]["wild_draw_four_penalty"] *= args.wild_risk_scale
        candidates.extend(sample_parameters(rng) for _ in range(args.candidates - 1))
    train_results = list(checkpoint.get("results", []))
    if len(train_results) > len(candidates):
        parser.error("checkpoint contains more candidates than this run")
    for index in range(len(train_results), len(candidates)):
        params = candidates[index]
        score = play_batch(strategy, params, args.games, args.seed, args.profile)
        train_results.append({"training": score, "parameters": params})
        checkpoint["results"] = train_results
        if args.checkpoint_file is not None:
            save_checkpoint(args.checkpoint_file, checkpoint)
        if (index + 1) % 10 == 0 or index + 1 == len(candidates):
            print(json.dumps({"training_progress": index + 1,
                              "of": len(candidates),
                              "best_points_so_far": max(
                                  row["training"]["average_points"]
                                  for row in train_results)}), flush=True)
    train_results.sort(key=lambda row: (
        row["training"]["average_points"], row["training"]["first_places"]
    ), reverse=True)

    validation_seed = args.seed + 1_000_000
    reference = checkpoint.get("reference")
    if reference is None:
        reference = play_batch(strategy, original_params, args.validation_games,
                               validation_seed, args.profile)
        checkpoint["reference"] = reference
        if args.checkpoint_file is not None:
            save_checkpoint(args.checkpoint_file, checkpoint)
    print(json.dumps({"reference_current_policy_same_seeds": reference,
                      "profile": args.profile}), flush=True)
    finalists = train_results[:min(args.top, len(train_results))]
    validated = list(checkpoint.get("validated", []))
    for rank, row in enumerate(finalists, start=1):
        if rank <= len(validated):
            continue
        score = play_batch(strategy, row["parameters"], args.validation_games,
                           validation_seed, args.profile)
        result = {"rank": rank, "validation": score,
                  "parameters": row["parameters"]}
        validated.append(result)
        checkpoint["validated"] = validated
        if args.checkpoint_file is not None:
            save_checkpoint(args.checkpoint_file, checkpoint)
        print(json.dumps(result), flush=True)

    best = max(validated, key=lambda row: row["validation"]["average_points"])
    print(json.dumps({"selected": best}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
