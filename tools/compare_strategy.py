#!/usr/bin/env python3
"""Run a paired fresh-seed comparison of tuned and current UNO weights."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_evaluator" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import tune_strategy  # noqa: E402
from triad_uno import engine, scoring, simulator  # noqa: E402


def game_points(strategy: Any, params: dict[str, float], seed: int,
                index: int, profile: str) -> tuple[float, bool]:
    strategy.PARAMS.clear()
    strategy.PARAMS.update(params)
    candidate_seat = index % engine.SEATS
    agents = []
    for seat in range(engine.SEATS):
        if seat == candidate_seat:
            agents.append(tune_strategy.CandidateAgent(strategy))
            continue
        kind = (
            tune_strategy.OPPONENTS[(index + seat + seed - index // engine.SEATS)
                                    % len(tune_strategy.OPPONENTS)]
            if profile == "mixed" else profile
        )
        agents.append(tune_strategy.make_opponent(kind, seed * 101 + seat))

    final, _, _ = simulator.run_game(seed, agents, engine.GameConfig())
    outcome = final.outcome
    if outcome is None:
        return 2.75, False
    outcome_dict = {"placements": outcome.placements,
                    "unfinished": outcome.unfinished}
    seat_names = ["candidate" if seat == candidate_seat
                  else f"opponent_{seat}" for seat in range(engine.SEATS)]
    points = scoring.points_for_game(outcome_dict, seat_names)["candidate"]
    return points, bool(outcome.placements
                        and outcome.placements[0] == candidate_seat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=3,
                        help="validation rank to compare from the tuning checkpoint")
    parser.add_argument("--games", type=int, default=10000,
                        help="paired games; use a multiple of four for seat rotation")
    parser.add_argument("--seed", type=int, default=2510000,
                        help="fresh seed block, disjoint from the tuning run")
    parser.add_argument("--profile", choices=("mixed", *tune_strategy.OPPONENTS),
                        default="mixed")
    args = parser.parse_args()
    if args.games < 4 or args.games % engine.SEATS:
        parser.error("games must be a positive multiple of four")

    checkpoint = json.loads(args.checkpoint_file.read_text(encoding="utf-8"))
    selected = next((row for row in checkpoint.get("validated", [])
                     if row.get("rank") == args.rank), None)
    if selected is None:
        parser.error(f"checkpoint has no validated candidate at rank {args.rank}")

    strategy = tune_strategy.load_strategy()
    current = dict(strategy.PARAMS)
    candidate = selected["parameters"]
    candidate_points: list[float] = []
    current_points: list[float] = []
    candidate_wins = 0
    current_wins = 0

    for index in range(args.games):
        seed = args.seed + index // engine.SEATS
        cand_score, cand_win = game_points(strategy, candidate, seed, index,
                                           args.profile)
        base_score, base_win = game_points(strategy, current, seed, index,
                                           args.profile)
        candidate_points.append(cand_score)
        current_points.append(base_score)
        candidate_wins += cand_win
        current_wins += base_win
        if (index + 1) % 2000 == 0 or index + 1 == args.games:
            print(json.dumps({"paired_progress": index + 1,
                              "of": args.games}), flush=True)

    deltas = [new - old for new, old in zip(candidate_points, current_points)]
    mean_delta = statistics.mean(deltas)
    standard_error = statistics.stdev(deltas) / (len(deltas) ** 0.5)
    result = {
        "games": args.games,
        "profile": args.profile,
        "candidate_rank": args.rank,
        "candidate_average_points": statistics.mean(candidate_points),
        "current_average_points": statistics.mean(current_points),
        "paired_mean_delta": mean_delta,
        "paired_95pct_ci": [mean_delta - 1.96 * standard_error,
                             mean_delta + 1.96 * standard_error],
        "candidate_first_places": candidate_wins,
        "current_first_places": current_wins,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
