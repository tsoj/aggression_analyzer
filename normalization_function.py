def normalize_raw_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize raw scores using statistical parameters: (value - mean) / std"""

    # Normalization parameters calculated from large dataset
    normalization_params = {
        "Sacrifice Score per Win Move": {"mean": 0.03224481, "std": 0.16040210},
        "Captures Near King": {"mean": 0.29434447, "std": 0.15217628},
        "Coordinated Attacks per Move": {"mean": 0.09329826, "std": 0.19610824},
        "Opposite-Side Castling Games": {"mean": 0.05882117, "std": 0.23529029},
        "Pawn Storms per Move": {"mean": 0.09775153, "std": 0.09111069},
        "Rook/Queen Threats per Move": {"mean": 0.03432748, "std": 0.04562306},
        "Moves Near King": {"mean": 0.22211190, "std": 0.23017310},
        "Advanced Pieces per Move": {"mean": 0.13637191, "std": 0.10244817},
        "Forcing Moves per Move": {"mean": 0.22818925, "std": 0.09221739},
        "Checks per Move": {"mean": 0.03978174, "std": 0.05454974},
        "Forfeited Castling Games": {"mean": 0.10092565, "std": 0.30123100},
        "Bishop/Queen Threats per Move": {"mean": 0.02520109, "std": 0.03944834},
        "Knight Outposts per Move": {"mean": 0.01250136, "std": 0.02492654},
        "Rook Lifts per Move": {"mean": 0.00898465, "std": 0.02311612},
        "Central Pawn Breaks per Move": {"mean": 0.02600630, "std": 0.03564559},
        "Short Game Bonus per Win": {"mean": 0.10775526, "std": 0.33671733},
        "F7/F2 Attacks per Move": {"mean": 0.01650602, "std": 0.02840060},
    }

    normalized_scores = {}
    for feature_name, raw_value in raw_scores.items():
        if feature_name in normalization_params:
            params = normalization_params[feature_name]
            if params['std'] > 0:
                # Normalize to mean=0, std=1
                normalized_scores[feature_name] = (raw_value - params['mean']) / params['std']
            else:
                normalized_scores[feature_name] = 0.0
        else:
            # Fallback for unknown features
            normalized_scores[feature_name] = raw_value

    return normalized_scores
