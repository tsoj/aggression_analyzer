# Chess Aggression Analysis Tool

A Python tool for analyzing aggressive play patterns in chess games.

## Files

- `aggression_analyzer.py` - Main analysis tool for calculating aggression scores from PGN files
- `tune_weights.py` - Optimize feature weights using labeled training data
- `calculate_normalization.py` - Calculate normalization parameters from game datasets

## Basic Usage

Analyze aggression for all players in a PGN file:
```bash
python aggression_analyzer.py --pgn games.pgn
```

Analyze a specific player:
```bash
python aggression_analyzer.py --pgn games.pgn --player "Magnus Carlsen"
```

## Requirements

- chess
- optuna (for weight tuning)
- zstandard (for compressed PGN files)
- tqdm