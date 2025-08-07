#!/usr/bin/env python3
"""
Script to calculate normalization parameters for raw aggression scores.
This analyzes a large set of games to determine mean and standard deviation
for each raw score feature, enabling better normalization.
"""

import argparse
import chess
import chess.pgn
import tempfile
import os
from typing import Dict, List
from collections import defaultdict
import statistics
import json

# Import the existing classes and functions
from main3_1 import AggressionStats, get_raw_feature_scores, analyse_game, download_and_decompress_pgn


def collect_raw_scores(pgn_path: str, max_games: int = None, min_rating: int = 2000) -> Dict[str, List[float]]:
    """
    Process PGN file and collect all raw scores for statistical analysis.

    Returns:
        Dictionary mapping feature names to lists of raw scores
    """
    raw_score_collections = defaultdict(list)
    games_processed = 0
    games_filtered_by_rating = 0

    print(f"Collecting raw scores from games (min rating: {min_rating})...")

    with open(pgn_path) as pgn_file:
        while True:
            if max_games and games_processed >= max_games:
                print(f"\nReached game limit of {max_games}.")
                break

            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
            except Exception:
                # Skip malformed games
                continue

            white_player = game.headers.get("White", "?")
            black_player = game.headers.get("Black", "?")
            if "?" in (white_player, black_player):
                continue

            # Filter by rating
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                min_elo = min(white_elo, black_elo)
                if min_elo > 0 and min_elo < min_rating:
                    games_filtered_by_rating += 1
                    continue
            except (ValueError, TypeError):
                continue

            # Analyze both players
            for player in [white_player, black_player]:
                stats = AggressionStats()
                analyse_game(game, player, stats)
                raw_scores = get_raw_feature_scores(stats)

                # Only collect scores if we have meaningful data
                if raw_scores and stats.num_games > 0:
                    for feature_name, score in raw_scores.items():
                        raw_score_collections[feature_name].append(score)

            games_processed += 1
            if games_processed % 100 == 0:
                print(f"\rProcessed {games_processed} games...", end="")

    print(f"\nProcessed {games_processed} games total.")
    if games_filtered_by_rating > 0:
        print(f"Filtered out {games_filtered_by_rating} games due to rating requirements.")

    return dict(raw_score_collections)


def calculate_normalization_parameters(raw_score_collections: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate mean and standard deviation for each feature.

    Returns:
        Dictionary mapping feature names to {'mean': float, 'std': float, 'count': int}
    """
    normalization_params = {}

    print(f"\nCalculating normalization parameters...")
    print(f"{'Feature':<35} {'Count':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 90)

    for feature_name, scores in raw_score_collections.items():
        if len(scores) < 10:  # Need at least 10 samples
            print(f"Warning: Only {len(scores)} samples for {feature_name}, skipping...")
            continue

        mean_val = statistics.mean(scores)
        std_val = statistics.stdev(scores) if len(scores) > 1 else 0.0
        min_val = min(scores)
        max_val = max(scores)

        normalization_params[feature_name] = {
            'mean': mean_val,
            'std': std_val,
            'count': len(scores),
            'min': min_val,
            'max': max_val
        }

        print(f"{feature_name:<35} {len(scores):<8} {mean_val:<12.6f} {std_val:<12.6f} {min_val:<12.6f} {max_val:<12.6f}")

    return normalization_params


def generate_normalization_code(normalization_params: Dict[str, Dict[str, float]]) -> str:
    """
    Generate Python code for the new normalization function.
    """
    code_lines = [
        "def normalize_raw_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:",
        '    """Normalize raw scores using statistical parameters: (value - mean) / std"""',
        "    ",
        "    # Normalization parameters calculated from large dataset",
        "    normalization_params = {"
    ]

    for feature_name, params in normalization_params.items():
        code_lines.append(f'        "{feature_name}": {{"mean": {params["mean"]:.8f}, "std": {params["std"]:.8f}}},')

    code_lines.extend([
        "    }",
        "    ",
        "    normalized_scores = {}",
        "    for feature_name, raw_value in raw_scores.items():",
        "        if feature_name in normalization_params:",
        "            params = normalization_params[feature_name]",
        "            if params['std'] > 0:",
        "                # Normalize to mean=0, std=1",
        "                normalized_scores[feature_name] = (raw_value - params['mean']) / params['std']",
        "            else:",
        "                normalized_scores[feature_name] = 0.0",
        "        else:",
        "            # Fallback for unknown features",
        "            normalized_scores[feature_name] = raw_value",
        "    ",
        "    return normalized_scores"
    ])

    return "\n".join(code_lines)


def main():
    parser = argparse.ArgumentParser(description="Calculate normalization parameters for chess aggression analysis.")
    parser.add_argument("--pgn", type=str, required=True, help="Path or URL to the PGN file (can be .pgn.zst).")
    parser.add_argument("--games", type=int, help="Maximum number of games to process.")
    parser.add_argument("--min_rating", type=int, default=2000, help="Minimum rating for the lower-rated player in each game (default: 2000).")
    parser.add_argument("--output", type=str, default="normalization_params.json", help="Output file for normalization parameters.")
    parser.add_argument("--code_output", type=str, default="normalization_function.py", help="Output file for generated normalization code.")

    args = parser.parse_args()

    # Handle URL downloads
    pgn_path = args.pgn
    tmpdir = None
    if pgn_path.startswith("http"):
        tmpdir = tempfile.TemporaryDirectory()
        local_pgn_path = os.path.join(tmpdir.name, "games.pgn")
        if not download_and_decompress_pgn(pgn_path, local_pgn_path):
            return
        pgn_path = local_pgn_path

    try:
        # Collect raw scores
        raw_score_collections = collect_raw_scores(pgn_path, args.games, args.min_rating)

        if not raw_score_collections:
            print("No raw scores collected. Check your PGN file and parameters.")
            return

        # Calculate normalization parameters
        normalization_params = calculate_normalization_parameters(raw_score_collections)

        # Save parameters as JSON
        with open(args.output, 'w') as f:
            json.dump(normalization_params, f, indent=2)
        print(f"\nNormalization parameters saved to: {args.output}")

        # Generate and save normalization code
        normalization_code = generate_normalization_code(normalization_params)
        with open(args.code_output, 'w') as f:
            f.write(normalization_code)
        print(f"Normalization function code saved to: {args.code_output}")

        print(f"\nSummary:")
        print(f"- Analyzed {sum(len(scores) for scores in raw_score_collections.values())} player-game combinations")
        print(f"- Calculated parameters for {len(normalization_params)} features")
        print(f"\nTo use the new normalization, replace the current normalization in get_aggression_score()")
        print(f"with the normalize_raw_scores() function from {args.code_output}")

    finally:
        if tmpdir:
            tmpdir.cleanup()


if __name__ == "__main__":
    main()
