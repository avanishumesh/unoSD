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
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_evaluator" / "src"))

from triad_uno import engine, scoring, simulator  # noqa: E402
from triad_uno.baselines import HeuristicAgent, RandomAgent  # noqa: E402


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


def make_opponent(kind: str, seed: int) -> Any:
    if kind == "random":
        return RandomAgent(seed)
    if kind == "first_legal":
        return FirstLegalAgent()
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
                 for name in ("heuristic", "random", "first_legal")}
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
                kind = ("heuristic", "random", "first_legal")[(index + seat + first_seed) % 3]
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


def main() -> int:
    """Run a seeded random search, then validate the strongest candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--games", type=int, default=100,
                        help="training games per parameter set; use a multiple of four")
    parser.add_argument("--validation-games", type=int, default=400,
                        help="fresh games for each finalist; use a multiple of four")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--search-seed", type=int, default=20261014)
    parser.add_argument("--profile", choices=("mixed", "heuristic", "random", "first_legal"),
                        default="mixed", help="opponents to optimize against")
    parser.add_argument("--wild-risk-scale", type=float, default=1.0,
                        help="scale Wild and Wild Draw Four penalties; below 1 is riskier")
    parser.add_argument("--params-file", type=Path,
                        help="evaluate these weights instead of searching (use --candidates 1)")
    args = parser.parse_args()
    if min(args.candidates, args.games, args.validation_games, args.top) < 1:
        parser.error("candidate, game, and finalist counts must be positive")
    if args.games % 4 or args.validation_games % 4:
        parser.error("games and validation-games must be multiples of four for seat rotation")
    if args.params_file is not None and args.candidates != 1:
        parser.error("--params-file requires --candidates 1")

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
    train_results = []
    for index, params in enumerate(candidates):
        score = play_batch(strategy, params, args.games, args.seed, args.profile)
        train_results.append({"training": score, "parameters": params})
        print(json.dumps({"candidate": index, **score}))
    train_results.sort(key=lambda row: row["training"]["average_points"], reverse=True)

    validation_seed = args.seed + 1_000_000
    reference = play_batch(strategy, original_params, args.validation_games, validation_seed,
                           args.profile)
    print(json.dumps({"reference_current_policy_same_seeds": reference,
                      "profile": args.profile}))
    finalists = train_results[:min(args.top, len(train_results))]
    validated = []
    for rank, row in enumerate(finalists, start=1):
        score = play_batch(strategy, row["parameters"], args.validation_games,
                           validation_seed, args.profile)
        result = {"rank": rank, "validation": score,
                  "parameters": row["parameters"]}
        validated.append(result)
        print(json.dumps(result))

    best = max(validated, key=lambda row: row["validation"]["average_points"])
    print(json.dumps({"selected": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
