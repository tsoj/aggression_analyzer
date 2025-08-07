# tune_weights.py

import os
import glob
import argparse
import chess.pgn
import optuna
from tqdm import tqdm
from typing import List, Dict, Tuple
import random

# Import the core logic from your original script
from aggression_analyzer import (
    AggressionStats,
    analyse_game,
    get_raw_feature_scores,
    normalization_params
)

def calculate_score_with_weights(raw_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculates a final aggression score given raw features and a set of weights.
    """
    if not raw_scores:
        return 0.0

    total_weighted_score = 0.0
    total_weight = 0.0

    for feature_name, raw_value in raw_scores.items():
        if feature_name not in weights:
            continue

        weight = weights[feature_name]

        if feature_name in normalization_params:
            params = normalization_params[feature_name]
            if params['std'] > 0:
                normalized_value = (raw_value - params['mean']) / params['std']
            else:
                normalized_value = 0.0
        else:
            normalized_value = raw_value

        total_weighted_score += weight * normalized_value
        total_weight += weight

    if total_weight == 0:
        return 0.0

    final_score = total_weighted_score / total_weight
    return final_score


def preprocess_games_from_folder(folder_path: str, target_label: float, max_games_per_class: int) -> List[Tuple[Dict[str, float], float]]:
    """
    Reads all PGNs in a folder, analyzes them, and returns a list of
    (raw_feature_scores, target_label) tuples.
    If target_label is 1.0 (attacking), only the WINNING player's stats are extracted.
    If target_label is 0.0 (normal), stats from BOTH players are extracted.
    """
    print(f"\nProcessing games from '{folder_path}' with target score {target_label}...")

    pgn_files = glob.glob(os.path.join(folder_path, '*.pgn'))
    if not pgn_files:
        print(f"Warning: No .pgn files found in {folder_path}")
        return []

    processed_data = []

    for pgn_path in tqdm(pgn_files, desc=f"Files in {os.path.basename(folder_path)}"):
        # Stop processing if we've hit the max limit for this class
        if len(processed_data) >= max_games_per_class:
            print(f"\nReached max_games_per_class limit of {max_games_per_class} for this set.")
            break

        try:
            with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                while True:
                    if len(processed_data) >= max_games_per_class:
                        break
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                    except (ValueError, IndexError):
                        continue

                    is_attacking_set = (target_label == 1.0)

                    if is_attacking_set:
                        # For attacking games, find the winner and analyze only their play.
                        result = game.headers.get("Result", "*")
                        winner_player_name = None
                        if result == "1-0":
                            winner_player_name = game.headers.get("White")
                        elif result == "0-1":
                            winner_player_name = game.headers.get("Black")

                        # If we found a clear winner, process their game. Otherwise, skip (e.g., draws).
                        if winner_player_name and winner_player_name != "?":
                            stats = AggressionStats()
                            analyse_game(game, winner_player_name, stats)
                            raw_scores = get_raw_feature_scores(stats)
                            if raw_scores:
                                processed_data.append((raw_scores, target_label))
                    else:
                        # For normal games, analyze both players as before.
                        players = {
                            "White": game.headers.get("White", "?"),
                            "Black": game.headers.get("Black", "?")
                        }
                        for _, player_name in players.items():
                            if player_name == "?" or len(processed_data) >= max_games_per_class:
                                continue

                            stats = AggressionStats()
                            analyse_game(game, player_name, stats)
                            raw_scores = get_raw_feature_scores(stats)

                            if raw_scores:
                                processed_data.append((raw_scores, target_label))
        except Exception as e:
            print(f"Could not process file {pgn_path}: {e}")

    # Shuffle the data before splitting to ensure the test set is random
    random.shuffle(processed_data)

    print(f"Found {len(processed_data)} valid player-perspectives in {folder_path}.")
    return processed_data


def objective(trial: optuna.trial.Trial, training_data: List[Tuple[Dict, float]]) -> float:
    """The objective function for Optuna to minimize."""
    weights = {}
    feature_names = normalization_params.keys()
    for feature in feature_names:
        weights[feature] = trial.suggest_float(feature, 0.0, 2.0)

    total_error = 0.0
    for raw_scores, target_label in training_data:
        predicted_score = calculate_score_with_weights(raw_scores, weights)
        total_error += abs(predicted_score - target_label)

    return total_error / len(training_data)


def evaluate_performance(dataset: List, weights: Dict, dataset_name: str):
    """Helper function to calculate and print performance metrics for a given dataset."""
    if not dataset:
        print(f"\n{dataset_name} is empty. Skipping evaluation.")
        return

    normal_scores = [calculate_score_with_weights(d[0], weights) for d in dataset if d[1] == 0.0]
    attacking_scores = [calculate_score_with_weights(d[0], weights) for d in dataset if d[1] == 1.0]

    print(f"\n--- {dataset_name} Performance ---")
    if normal_scores:
        avg_normal = sum(normal_scores) / len(normal_scores)
        print(f"Average score for 'normal' games:   {avg_normal:.4f} (Target: 0.0)")
    else:
        print("No 'normal' games in this set.")

    if attacking_scores:
        avg_attacking = sum(attacking_scores) / len(attacking_scores)
        print(f"Average score for 'attacking' games: {avg_attacking:.4f} (Target: 1.0)")
    else:
        print("No 'attacking' games in this set.")


def create_train_test_split(data: List, test_split: float) -> Tuple[List, List]:
    """Create train/test split. If test_split is 0.0, returns all data as training set."""
    if test_split == 0.0:
        return data, []

    split_idx = int(len(data) * (1.0 - test_split))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Tune aggression feature weights using Optuna.")
    parser.add_argument("--normal_games_dir", type=str, required=True, help="Path to folder with 'normal' PGNs (target=0.0).")
    parser.add_argument("--attacking_games_dir", type=str, required=True, help="Path to folder with 'attacking' PGNs (target=1.0).")
    parser.add_argument("--max_games_per_class", type=int, default=12_000, help="Maximum number of games to load for each class.")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials to run.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of data to use for testing (0.0 = no test set, 0.1 = 10%% test set).")
    args = parser.parse_args()

    # --- 1. Pre-process all data ---
    normal_data = preprocess_games_from_folder(args.normal_games_dir, 0.0, args.max_games_per_class)
    attacking_data = preprocess_games_from_folder(args.attacking_games_dir, 1.0, args.max_games_per_class)

    # --- 2. Create Train/Test splits ---
    if args.test_split == 0.0:
        print("\nNo test set will be created (test_split=0.0). All data will be used for training.")
    else:
        print(f"\nCreating {int((1.0 - args.test_split) * 100)}/{int(args.test_split * 100)} train-test splits...")

    # Split normal data
    normal_train, normal_test = create_train_test_split(normal_data, args.test_split)

    # Split attacking data
    attacking_train, attacking_test = create_train_test_split(attacking_data, args.test_split)

    # Combine to create final datasets
    all_training_data = normal_train + attacking_train
    all_testing_data = normal_test + attacking_test

    # Shuffle the combined training data one more time
    random.shuffle(all_training_data)

    print(f"Total training examples: {len(all_training_data)} ({len(normal_train)} normal, {len(attacking_train)} attacking)")
    if args.test_split > 0.0:
        print(f"Total testing examples:  {len(all_testing_data)} ({len(normal_test)} normal, {len(attacking_test)} attacking)")
    else:
        print("Total testing examples:  0 (no test set)")

    if not all_training_data:
        print("\nError: Training data is empty. Cannot proceed with optimization.")
        return

    # --- 3. Run the optimization study on the TRAINING set ---
    study = optuna.create_study(direction='minimize')
    print(f"\nStarting optimization with {args.trials} trials on the training set...")
    study.optimize(
        lambda trial: objective(trial, all_training_data),
        n_trials=args.trials,
        show_progress_bar=True
    )

    # --- 4. Display the results ---
    best_weights = study.best_params
    best_value = study.best_value

    print("\n\n--- Optimization Complete ---")
    print(f"Best Mean Squared Error on Training Set: {best_value:.6f}")
    print("\nOptimized Feature Weights (copy this into your `aggression_analyzer.py`):")
    print("-" * 70)
    print("feature_weights = {")
    for name, weight in sorted(best_weights.items()):
        print(f'    "{name}": {weight:.6f},')
    print("}")
    print("-" * 70)

    # --- 5. Evaluate final model on train set (and test set if available) ---
    evaluate_performance(all_training_data, best_weights, "Training Set")
    if args.test_split > 0.0:
        evaluate_performance(all_testing_data, best_weights, "Test Set")

if __name__ == "__main__":
    main()
