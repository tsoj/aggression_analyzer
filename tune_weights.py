# tune_weights.py

import os
import glob
import argparse
import chess.pgn
import optuna
from tqdm import tqdm
from typing import List, Dict, Tuple

# Import the core logic from your original script
from aggression_analyzer import (
    AggressionStats,
    analyse_game,
    get_raw_feature_scores,
    normalization_params  # We need the normalization constants
)

def calculate_score_with_weights(raw_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculates a final aggression score given raw features and a set of weights.
    This is a modified version of the original get_aggression_score function.
    """
    if not raw_scores:
        return 0.0

    total_weighted_score = 0.0
    total_weight = 0.0

    for feature_name, raw_value in raw_scores.items():
        # Ensure we only use features we are optimizing
        if feature_name not in weights:
            continue

        weight = weights[feature_name]

        # Normalize the raw value using the pre-calculated stats
        if feature_name in normalization_params:
            params = normalization_params[feature_name]
            if params['std'] > 0:
                normalized_value = (raw_value - params['mean']) / params['std']
            else:
                normalized_value = 0.0
        else:
            # Fallback for features not in our normalization map
            normalized_value = raw_value

        total_weighted_score += weight * normalized_value
        total_weight += weight

    if total_weight == 0:
        return 0.0

    # The final score is the average weighted normalized value.
    # This can be positive or negative.
    final_score = total_weighted_score / total_weight
    return final_score


def preprocess_games_from_folder(folder_path: str, target_label: float, max_games_per_class: int) -> List[Tuple[Dict[str, float], float]]:
    """
    Reads all PGNs in a folder, analyzes them, and returns a list of
    (raw_feature_scores, target_label) tuples.
    """
    print(f"\nProcessing games from '{folder_path}' with target score {target_label}...")

    pgn_files = glob.glob(os.path.join(folder_path, '*.pgn'))
    if not pgn_files:
        print(f"Warning: No .pgn files found in {folder_path}")
        return []

    processed_data = []

    for pgn_path in tqdm(pgn_files, desc=f"Files in {os.path.basename(folder_path)}"):
        try:
            with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                while True:
                    if len(processed_data) > max_games_per_class:
                        break
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                    except (ValueError, IndexError):
                        # Skip malformed games
                        continue

                    players = {
                        "White": game.headers.get("White", "?"),
                        "Black": game.headers.get("Black", "?")
                    }

                    # A single game provides two data points (one for each player)
                    for color, player_name in players.items():
                        if player_name == "?":
                            continue

                        # Analyze the game from this player's perspective
                        stats = AggressionStats()
                        analyse_game(game, player_name, stats)

                        # Get the raw, unweighted feature scores
                        raw_scores = get_raw_feature_scores(stats)

                        if raw_scores:
                            processed_data.append((raw_scores, target_label))
        except Exception as e:
            print(f"Could not process file {pgn_path}: {e}")

    print(f"Found {len(processed_data)} valid player-games in {folder_path}.")
    return processed_data


def objective(trial: optuna.trial.Trial, training_data: List[Tuple[Dict, float]]) -> float:
    """
    The objective function for Optuna to minimize.
    """
    # Define the search space for the weights.
    # We suggest a new weight for each feature in each trial.
    # A range of 0.0 to 2.0 is a good starting point. 0 disables a feature,
    # 1 is a neutral baseline, >1 gives it more importance.
    weights = {}
    feature_names = normalization_params.keys()
    for feature in feature_names:
        weights[feature] = trial.suggest_float(feature, 0.0, 2.0)

    total_squared_error = 0.0

    # Calculate the error for the current set of weights
    for raw_scores, target_label in training_data:
        predicted_score = calculate_score_with_weights(raw_scores, weights)
        error = predicted_score - target_label
        total_squared_error += error * error

    # Return the Mean Squared Error
    return total_squared_error / len(training_data)


def main():
    parser = argparse.ArgumentParser(description="Tune aggression feature weights using Optuna.")
    parser.add_argument("--normal_games_dir", type=str, required=True, help="Path to folder with 'normal' PGNs (target=0.0).")
    parser.add_argument("--attacking_games_dir", type=str, required=True, help="Path to folder with 'attacking' PGNs (target=1.0).")
    parser.add_argument("--max_games_per_class", type=int, default=10_000, help="maximum number of games to have in each game class.")
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials to run.")
    args = parser.parse_args()

    # --- 1. Pre-process all data ---
    # This is done only ONCE to save time.
    normal_data = preprocess_games_from_folder(args.normal_games_dir, target_label=0.0, max_games_per_class=max_games_per_class)
    attacking_data = preprocess_games_from_folder(args.attacking_games_dir, target_label=1.0, max_games_per_class=max_games_per_class)

    all_training_data = normal_data + attacking_data

    if not all_training_data:
        print("Error: No game data could be loaded. Please check your PGN folders.")
        return

    # --- 2. Run the optimization study ---
    study = optuna.create_study(direction='minimize')
    print(f"\nStarting optimization with {args.trials} trials...")
    study.optimize(
        lambda trial: objective(trial, all_training_data),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # --- 3. Display the results ---
    best_weights = study.best_params
    best_value = study.best_value

    print("\n\n--- Optimization Complete ---")
    print(f"Best Mean Squared Error: {best_value:.6f}")
    print("\nOptimized Feature Weights (copy this into your `aggression_analyzer.py`):")
    print("-" * 70)
    print("feature_weights = {")
    for name, weight in sorted(best_weights.items()):
        print(f'    "{name}": {weight:.6f},')
    print("}")
    print("-" * 70)

    # --- 4. (Optional) Sanity check the results ---
    print("\nVerifying performance with new weights...")
    normal_scores = [calculate_score_with_weights(d[0], best_weights) for d in normal_data]
    attacking_scores = [calculate_score_with_weights(d[0], best_weights) for d in attacking_data]

    if normal_scores:
        avg_normal = sum(normal_scores) / len(normal_scores)
        print(f"Average score for 'normal' games:   {avg_normal:.4f} (Target: 0.0)")
    if attacking_scores:
        avg_attacking = sum(attacking_scores) / len(attacking_scores)
        print(f"Average score for 'attacking' games: {avg_attacking:.4f} (Target: 1.0)")


if __name__ == "__main__":
    main()
