import time
import chess
import chess.pgn
import argparse
from dataclasses import dataclass, field
from typing import List

# --- Configuration ---
# A material advantage greater than this value means a player is already
# considered to be in a "winning" position. Sacrifices are not counted
# when the player is this far ahead.
WINNING_MATERIAL_ADVANTAGE = 4
# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

@dataclass
class AggressionStats:
    """
    Stores all statistics needed to calculate an aggression score.
    """
    # Game outcomes
    num_games: int = 0
    num_wins: int = 0
    num_draws: int = 0
    num_losses: int = 0
    total_moves: int = 0

    # 1. King Position & Safety
    opposite_side_castling_games: int = 0
    forfeited_castling_games: int = 0 # King remains in center past move 20
    pawn_storms_vs_king: int = 0
    total_checks: int = 0
    # NEW: Distance-based king attacks
    capture_distance_from_king: list[int] = field(default_factory=lambda: [0] * 8)
    move_distance_from_king: list[int] = field(default_factory=lambda: [0] * 8)
    # NEW: Threats on king's lines
    rook_queen_threats: int = 0
    bishop_queen_threats: int = 0

    # 2. Pawn Structure & Line Opening
    central_pawn_breaks: int = 0

    # 3. Piece Activity & Placement
    advanced_pieces: int = 0
    rook_lifts: int = 0
    knight_outposts: int = 0

    # 4. Move Characteristics & Material Balance
    material_sacrifices: int = 0 # Count of times a deficit was initiated
    exchange_sacrifices: int = 0
    # NEW: Advanced sacrifice metric
    total_plys_in_sacrifice: int = 0 # Total plys spent in a material deficit in won games
    forcing_moves: int = 0

    # 5. Tactical Patterns
    f7_f2_attacks: int = 0
    coordinated_attacks: int = 0

def get_material_score(board: chess.Board, side: chess.Color) -> int:
    """Calculates the material value for a given side."""
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, side)) * value
    return score

def analyse_game(game: chess.pgn.Game, player_name: str, stats: AggressionStats):
    """
    Analyzes a single game for the specified player and updates the stats object.
    """
    player_color = chess.WHITE if game.headers["White"] == player_name else chess.BLACK

    board = game.board()
    us_castled_side = None
    them_castled_side = None

    is_win = (game.headers["Result"] == "1-0" and player_color == chess.WHITE) or \
             (game.headers["Result"] == "0-1" and player_color == chess.BLACK)

    # State for advanced sacrifice detection
    plys_in_current_sacrifice = 0

    for ply, move in enumerate(game.mainline_moves()):
        is_player_turn = (board.turn == player_color)

        # --- Sacrifice Duration Tracking (checked at the start of our turn) ---
        if is_player_turn and is_win:
            material_us = get_material_score(board, player_color)
            material_them = get_material_score(board, not player_color)
            material_advantage = material_us - material_them

            # Are we currently in a sacrificed state?
            if material_advantage < 0 and material_advantage > -WINNING_MATERIAL_ADVANTAGE:
                plys_in_current_sacrifice += 1
            # Did the sacrifice just end?
            elif plys_in_current_sacrifice > 0:
                stats.total_plys_in_sacrifice += plys_in_current_sacrifice
                plys_in_current_sacrifice = 0

        # --- Analysis for the current move (if it's our turn) ---
        if is_player_turn:
            stats.total_moves += 1
            material_us_before = get_material_score(board, player_color)
            material_them_before = get_material_score(board, not player_color)
            material_advantage = material_us_before - material_them_before
            enemy_king_square = board.king(not player_color)
            moving_piece_type = board.piece_type_at(move.from_square)

            # King Proximity & Threats
            dist_to_king = chess.square_distance(move.to_square, enemy_king_square)
            if board.is_capture(move):
                stats.capture_distance_from_king[dist_to_king] += 1
            else:
                stats.move_distance_from_king[dist_to_king] += 1

            dx = chess.square_file(move.to_square) - chess.square_file(enemy_king_square)
            dy = chess.square_rank(move.to_square) - chess.square_rank(enemy_king_square)
            if moving_piece_type in [chess.ROOK, chess.QUEEN] and (dx == 0 or dy == 0):
                stats.rook_queen_threats += 1
            if moving_piece_type in [chess.BISHOP, chess.QUEEN] and abs(dx) == abs(dy):
                stats.bishop_queen_threats += 1

            # (The rest of the features from the previous script)
            # ... [Code for pawn storms, central breaks, piece activity, etc.]
            # This part is largely unchanged but integrated into the single loop.
            if board.is_castling(move):
                us_castled_side = "K" if board.is_kingside_castling(move) else "Q"
                if them_castled_side and us_castled_side != them_castled_side:
                    stats.opposite_side_castling_games += 1
            if moving_piece_type == chess.PAWN and them_castled_side and abs(chess.square_file(move.from_square) - chess.square_file(enemy_king_square)) <= 2:
                stats.pawn_storms_vs_king += 1
            if moving_piece_type == chess.PAWN and chess.square_file(move.from_square) in [3, 4]:
                if (player_color == chess.WHITE and chess.square_rank(move.to_square) == 4) or (player_color == chess.BLACK and chess.square_rank(move.to_square) == 3):
                    stats.central_pawn_breaks += 1
            if moving_piece_type != chess.PAWN and ((player_color == chess.WHITE and chess.square_rank(move.to_square) >= 4) or (player_color == chess.BLACK and chess.square_rank(move.to_square) <= 3)):
                stats.advanced_pieces += 1
            if moving_piece_type == chess.ROOK:
                if (player_color == chess.WHITE and chess.square_rank(move.from_square) in [0, 1] and chess.square_rank(move.to_square) == 2) or (player_color == chess.BLACK and chess.square_rank(move.from_square) in [7, 6] and chess.square_rank(move.to_square) == 5):
                    stats.rook_lifts += 1
            if moving_piece_type == chess.KNIGHT and ((player_color == chess.WHITE and chess.square_rank(move.to_square) >= 4) or (player_color == chess.BLACK and chess.square_rank(move.to_square) <= 3)):
                if any(board.piece_type_at(sq) == chess.PAWN for sq in board.attackers(player_color, move.to_square)):
                    stats.knight_outposts += 1
            is_capture = board.is_capture(move)
            is_check = board.gives_check(move)
            if is_capture or is_check: stats.forcing_moves += 1
            if is_check: stats.total_checks += 1
            target_square = chess.F7 if player_color == chess.WHITE else chess.F2
            if move.to_square == target_square or target_square in board.attacks(move.to_square): stats.f7_f2_attacks += 1
            if len(board.attackers(player_color, enemy_king_square)) >= 3: stats.coordinated_attacks += 1

            # Simplified Sacrifice Initiation check
            if is_win and material_advantage < WINNING_MATERIAL_ADVANTAGE:
                board.push(move)
                mat_us_after = get_material_score(board, player_color)
                mat_them_after = get_material_score(board, not player_color)
                if mat_us_after < mat_them_after and material_advantage >= 0:
                    stats.material_sacrifices += 1 # A sac was initiated
                    if is_capture and moving_piece_type == chess.ROOK and board.piece_type_at(move.to_square) in [chess.BISHOP, chess.KNIGHT]:
                        stats.exchange_sacrifices += 1
                board.pop()

        else: # Opponent's turn
            if board.is_castling(move):
                them_castled_side = "K" if board.is_kingside_castling(move) else "Q"

        board.push(move)

    # Post-game analysis
    if plys_in_current_sacrifice > 0: # Add any sacrifice duration that lasted until game end
        stats.total_plys_in_sacrifice += plys_in_current_sacrifice
    if not us_castled_side and ply >= 40:
        stats.forfeited_castling_games += 1

    stats.num_games += 1
    if is_win:
        stats.num_wins += 1
    elif game.headers["Result"] == "1/2-1/2":
        stats.num_draws += 1
    else:
        stats.num_losses += 1


def get_aggression_score(stats: AggressionStats, verbose: bool = False) -> float:
    if stats.num_games == 0 or stats.total_moves == 0:
        return 0.0

    # --- Helper functions for complex feature scoring ---
    def score_near_king(distance_list: list[int], total_events: int) -> float:
        if total_events == 0: return 0.0
        weights = [0, 8, 4, 2, 1, 0, 0, 0] # dist 0 is impossible, dist 1 is max score
        score = sum(weights[dist] * freq for dist, freq in enumerate(distance_list))
        max_score = max(weights) * total_events
        return score / max_score if max_score > 0 else 0.0

    total_captures = sum(stats.capture_distance_from_king)
    total_non_captures = sum(stats.move_distance_from_king)

    # --- Feature Definition: (Weight, Name, Value) ---
    features = [
        # --- TOP TIER: Direct, high-risk, undeniable aggression ---
        (20.0, "Sacrifice Duration per Win", stats.total_plys_in_sacrifice / stats.num_wins if stats.num_wins > 0 else 0),
        (12.0, "Captures Near King", score_near_king(stats.capture_distance_from_king, total_captures)),
        (10.0, "Opposite-Side Castling", stats.opposite_side_castling_games / stats.num_games),

        # --- HIGH TIER: Strong indicators of attacking intent ---
        (8.0, "Moves Near King", score_near_king(stats.move_distance_from_king, total_non_captures)),
        (8.0, "Rook/Queen Threats on King", stats.rook_queen_threats / stats.total_moves),
        (8.0, "Pawn Storms per Move", stats.pawn_storms_vs_king / stats.total_moves),
        (7.0, "Coordinated Attacks per Move", stats.coordinated_attacks / stats.total_moves),

        # --- MID TIER: Common traits of aggressive players ---
        (6.0, "Forcing Moves Ratio", stats.forcing_moves / stats.total_moves),
        (6.0, "Bishop/Queen Threats on King", stats.bishop_queen_threats / stats.total_moves),
        (5.0, "Advanced Piece Placement", stats.advanced_pieces / stats.total_moves),
        (5.0, "Forfeited Castling", stats.forfeited_castling_games / stats.num_games),

        # --- CONTRIBUTING FACTORS ---
        (4.0, "Knight Outposts per Move", stats.knight_outposts / stats.total_moves),
        (3.0, "Exchange Sacrifices per Win", stats.exchange_sacrifices / stats.num_wins if stats.num_wins > 0 else 0),
        (3.0, "Rook Lifts per Move", stats.rook_lifts / stats.total_moves),
    ]

    normalization_caps = {
        "Sacrifice Duration per Win": 10.0, # Avg 10 plys in sacrifice per win is extremely high
        "Captures Near King": 1.0,         # Already normalized by helper
        "Opposite-Side Castling": 0.5,
        "Moves Near King": 1.0,            # Already normalized by helper
        "Rook/Queen Threats on King": 0.05,
        "Pawn Storms per Move": 0.1,
        "Coordinated Attacks per Move": 0.05,
        "Forcing Moves Ratio": 0.4,
        "Bishop/Queen Threats on King": 0.05,
        "Advanced Piece Placement": 0.2,
        "Forfeited Castling": 0.3,
        "Knight Outposts per Move": 0.05,
        "Exchange Sacrifices per Win": 0.1,
        "Rook Lifts per Move": 0.02,
    }

    # --- Calculation (same logic as before) ---
    total_score = 0.0
    total_weight = 0.0

    if verbose:
        print("\n--- Aggression Score Breakdown ---")
        print(f"{'Feature':<30} | {'Raw Value':>10} | {'Norm Value':>10} | {'Weighted':>10} | {'Contrib%':>10}")
        print("-" * 78)

    weighted_scores = []
    for weight, name, value in features:
        cap = normalization_caps.get(name, 1.0)
        normalized_value = min(value / cap, 1.0) if cap > 0 else 0
        weighted_score = weight * normalized_value
        weighted_scores.append(weighted_score)
        total_weight += weight

    total_weighted_score = sum(weighted_scores)

    if verbose and total_weighted_score > 0:
        for i, (weight, name, value) in enumerate(features):
            contribution = (weighted_scores[i] / total_weighted_score * 100)
            cap = normalization_caps.get(name, 1.0)
            normalized_value = min(value / cap, 1.0) if cap > 0 else 0
            print(f"{name:<30} | {value:10.4f} | {normalized_value:10.4f} | {weighted_scores[i]:10.4f} | {contribution:9.2f}%")

    final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    if verbose: print("-" * 78)
    return final_score


def main():
    parser = argparse.ArgumentParser(description="Advanced Chess Aggression Analysis Tool")
    parser.add_argument("--pgn", type=str, required=True, help="Path to the PGN file.")
    parser.add_argument("--player", type=str, required=True, help="Name of the player to analyze.")
    parser.add_argument("--games", type=int, help="Maximum number of games to analyze.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown of the aggression score.")
    args = parser.parse_args()

    stats = AggressionStats()
    games_analyzed = 0

    with open(args.pgn) as pgn:
        while True:
            if args.games and games_analyzed >= args.games:
                print(f"\nReached game limit of {args.games}.")
                break

            headers = chess.pgn.read_headers(pgn)
            if headers is None: break

            if args.player in (headers.get("White", "?"), headers.get("Black", "?")):
                game = chess.pgn.game_from_headers(headers)
                try:
                    analyse_game(game, args.player, stats)
                    games_analyzed += 1
                    print(f"\rAnalyzed {games_analyzed} games...", end="")
                except Exception as e:
                    print(f"\nSkipping a game due to an error: {e}")

    print("\n\n--- Analysis Complete ---")
    print(f"Player: {args.player}")

    if stats.num_games > 0:
        print("\nOverall Stats:")
        print(f"- Games Analyzed: {stats.num_games}")
        print(f"- Wins:   {stats.num_wins} ({100*stats.num_wins/stats.num_games:.1f}%)")
        print(f"- Draws:  {stats.num_draws} ({100*stats.num_draws/stats.num_games:.1f}%)")
        print(f"- Losses: {stats.num_losses} ({100*stats.num_losses/stats.num_games:.1f}%)")

        score = get_aggression_score(stats, args.verbose)

        print("\n--- Final Score ---")
        print(f"Aggression Score: {score:.4f}")
        print("(A score from 0.0 to 1.0 indicating aggressive tendencies)")
    else:
        print("No games found for this player in the PGN file.")

if __name__ == '__main__':
    main()
