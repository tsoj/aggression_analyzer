def normalize_raw_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize raw scores using statistical parameters: (value - mean) / std"""
    
    # Normalization parameters calculated from large dataset
    normalization_params = {
        "Sacrifice Score per Win Move": {"mean": 0.04547650, "std": 0.18451553},
        "Captures Near King": {"mean": 0.29840498, "std": 0.15439151},
        "Coordinated Attacks per Move": {"mean": 0.02644370, "std": 0.06003311},
        "Opposite-Side Castling Games": {"mean": 0.05628226, "std": 0.23047176},
        "Pawn Storms per Move": {"mean": 0.09178491, "std": 0.09173833},
        "Rook/Queen Threats per Move": {"mean": 0.03563800, "std": 0.04836624},
        "Moves Near King": {"mean": 0.22080340, "std": 0.22682390},
        "Advanced Pieces per Move": {"mean": 0.14197955, "std": 0.10708542},
        "Forcing Moves per Move": {"mean": 0.23460230, "std": 0.09857637},
        "Checks per Move": {"mean": 0.04440237, "std": 0.06128079},
        "Forfeited Castling Games": {"mean": 0.10098732, "std": 0.30131948},
        "Bishop/Queen Threats per Move": {"mean": 0.03001070, "std": 0.04570639},
        "Knight Outposts per Move": {"mean": 0.01248724, "std": 0.02620040},
        "Rook Lifts per Move": {"mean": 0.00826435, "std": 0.02250758},
        "Central Pawn Breaks per Move": {"mean": 0.02958451, "std": 0.04012492},
        "Short Game Bonus per Win": {"mean": 0.16011293, "std": 0.40578258},
        "F7/F2 Attacks per Move": {"mean": 0.01747349, "std": 0.03140933},
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