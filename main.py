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

@dataclass
class AggressionStats:
    """
    Stores all statistics needed to calculate an aggression score.
    Each statistic is tracked on a per-game basis and then aggregated.
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
    pawn_storms_vs_king: int = 0 # g, h, or f-pawn pushes against castled king
    total_checks: int = 0

    # 2. Pawn Structure & Line Opening
    central_pawn_breaks: int = 0 # d4-d5, e4-e5 pushes

    # 3. Piece Activity & Placement
    advanced_pieces: int = 0 # Pieces on 5th/6th rank (White) or 4th/3rd (Black)
    rook_lifts: int = 0
    knight_outposts: int = 0

    # 4. Move Characteristics & Material Balance
    material_sacrifices: int = 0 # Voluntarily giving up material in non-losing, won games
    exchange_sacrifices: int = 0
    forcing_moves: int = 0 # Captures + Checks

    # 5. Tactical Patterns
    f7_f2_attacks: int = 0 # Pieces attacking the weak f-pawn square
    coordinated_attacks: int = 0 # 3+ pieces attacking the king zone

def get_material_score(board: chess.Board, side: chess.Color) -> int:
    """Calculates the material value for a given side."""
    score = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        value = chess.piece_values[piece_type]
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

    # --- Pre-computation for the game ---
    is_win = (game.headers["Result"] == "1-0" and player_color == chess.WHITE) or \
             (game.headers["Result"] == "0-1" and player_color == chess.BLACK)

    for ply, move in enumerate(game.mainline_moves()):
        if board.turn != player_color:
            # Track opponent's castling
            if board.is_castling(move):
                them_castled_side = "K" if board.is_kingside_castling(move) else "Q"
            board.push(move)
            continue

        # --- Analysis is done *before* the move is made ---
        material_us_before = get_material_score(board, player_color)
        material_them_before = get_material_score(board, not player_color)
        material_advantage = material_us_before - material_them_before

        enemy_king_square = board.king(not player_color)
        moving_piece_type = board.piece_type_at(move.from_square)

        # 1. King Safety Indicators
        if board.is_castling(move):
            us_castled_side = "K" if board.is_kingside_castling(move) else "Q"
            if them_castled_side and us_castled_side != them_castled_side:
                stats.opposite_side_castling_games += 1
        
        # Pawn Storms
        if moving_piece_type == chess.PAWN and them_castled_side:
            pawn_file = chess.square_file(move.from_square)
            king_file = chess.square_file(enemy_king_square)
            if abs(pawn_file - king_file) <= 2: # Pawn is on or near the king's file
                stats.pawn_storms_vs_king += 1

        # 2. Pawn Structure
        if moving_piece_type == chess.PAWN:
            from_file = chess.square_file(move.from_square)
            if from_file in [3, 4]: # d or e file
                to_rank = chess.square_rank(move.to_square)
                if (player_color == chess.WHITE and to_rank == 4) or \
                   (player_color == chess.BLACK and to_rank == 3):
                    stats.central_pawn_breaks += 1
        
        # 3. Piece Activity
        to_rank = chess.square_rank(move.to_square)
        advanced_rank_threshold = 4 if player_color == chess.WHITE else 3
        if moving_piece_type != chess.PAWN and to_rank >= advanced_rank_threshold:
            stats.advanced_pieces += 1
        
        if moving_piece_type == chess.ROOK:
            from_rank = chess.square_rank(move.from_square)
            if (player_color == chess.WHITE and from_rank in [0, 1] and to_rank == 2) or \
               (player_color == chess.BLACK and from_rank in [7, 6] and to_rank == 5):
                stats.rook_lifts += 1
        
        if moving_piece_type == chess.KNIGHT:
            # Simple check for knight outpost: supported by pawn, deep in enemy territory
            if (player_color == chess.WHITE and to_rank >= 4) or \
               (player_color == chess.BLACK and to_rank <= 3):
                # Check for friendly pawn support
                support_squares = board.attackers(player_color, move.to_square)
                if any(board.piece_type_at(sq) == chess.PAWN for sq in support_squares):
                    stats.knight_outposts += 1

        # 4. Move Characteristics & Material Balance
        is_capture = board.is_capture(move)
        is_check = board.gives_check(move)

        if is_capture or is_check:
            stats.forcing_moves += 1
        if is_check:
            stats.total_checks += 1
        
        # Sacrifice Detection (only for wins, and not when already winning)
        if is_win and material_advantage < WINNING_MATERIAL_ADVANTAGE:
            board.push(move)
            material_us_after = get_material_score(board, player_color)
            if material_us_after < material_us_before:
                stats.material_sacrifices += 1
                if is_capture and moving_piece_type == chess.ROOK and \
                   board.piece_type_at(move.to_square) in [chess.BISHOP, chess.KNIGHT]:
                    stats.exchange_sacrifices += 1
            board.pop() # Revert move for next iteration

        # 5. Tactical Patterns
        # F7/F2 Attacks
        target_square = chess.F7 if player_color == chess.WHITE else chess.F2
        if move.to_square == target_square or target_square in board.attacks(move.to_square):
            stats.f7_f2_attacks += 1

        # Coordinated Attack (3+ pieces attacking the king zone)
        attackers = board.attackers(player_color, enemy_king_square)
        if len(attackers) >= 3:
            stats.coordinated_attacks += 1
        
        board.push(move)
        stats.total_moves += 1

    # --- Post-game analysis ---
    if not us_castled_side and ply >= 40: # 20 full moves
        stats.forfeited_castling_games += 1
        
    stats.num_games += 1
    if is_win:
        stats.num_wins += 1
    elif game.headers["Result"] == "1/2-1/2":
        stats.num_draws += 1
    else:
        stats.num_losses += 1


def get_aggression_score(stats: AggressionStats, verbose: bool = False) -> float:
    """
    Calculates a single aggression score based on the collected statistics.
    The score is a weighted average of normalized feature values.
    """
    if stats.num_games == 0:
        return 0.0

    # Each feature is a tuple: (Weight, Name, Value)
    # The value is normalized per game or per move to be comparable.
    features = [
        # --- High-impact indicators (strong signals of aggression) ---
        (15.0, "Material Sacrifices per Win", stats.material_sacrifices / stats.num_wins if stats.num_wins > 0 else 0),
        (12.0, "Opposite-Side Castling Games", stats.opposite_side_castling_games / stats.num_games),
        (10.0, "Coordinated Attacks per Move", stats.coordinated_attacks / stats.total_moves if stats.total_moves > 0 else 0),
        (8.0, "Exchange Sacrifices per Win", stats.exchange_sacrifices / stats.num_wins if stats.num_wins > 0 else 0),
        (8.0, "Pawn Storms per Move", stats.pawn_storms_vs_king / stats.total_moves if stats.total_moves > 0 else 0),

        # --- Medium-impact indicators (common in aggressive play) ---
        (6.0, "Forcing Moves per Move", stats.forcing_moves / stats.total_moves if stats.total_moves > 0 else 0),
        (6.0, "Advanced Pieces per Move", stats.advanced_pieces / stats.total_moves if stats.total_moves > 0 else 0),
        (5.0, "Forfeited Castling Games", stats.forfeited_castling_games / stats.num_games),
        (5.0, "Checks per Move", stats.total_checks / stats.total_moves if stats.total_moves > 0 else 0),
        
        # --- Lower-impact indicators (contributing factors) ---
        (4.0, "Knight Outposts per Move", stats.knight_outposts / stats.total_moves if stats.total_moves > 0 else 0),
        (3.0, "Rook Lifts per Move", stats.rook_lifts / stats.total_moves if stats.total_moves > 0 else 0),
        (3.0, "Central Pawn Breaks per Move", stats.central_pawn_breaks / stats.total_moves if stats.total_moves > 0 else 0),
        (2.0, "F7/F2 Attacks per Move", stats.f7_f2_attacks / stats.total_moves if stats.total_moves > 0 else 0),
    ]

    total_score = 0.0
    total_weight = 0.0
    
    # Cap each feature's raw value to prevent outliers from dominating the score.
    # This is a simple way to approximate normalization to a [0, 1] range.
    # For example, 0.1 means we don't expect this feature to occur more than 10% of the time.
    normalization_caps = {
        "Material Sacrifices per Win": 0.2,
        "Opposite-Side Castling Games": 0.5,
        "Coordinated Attacks per Move": 0.05,
        "Exchange Sacrifices per Win": 0.1,
        "Pawn Storms per Move": 0.1,
        "Forcing Moves per Move": 0.4, # Captures/checks are common
        "Advanced Pieces per Move": 0.2,
        "Forfeited Castling Games": 0.3,
        "Checks per Move": 0.1,
        "Knight Outposts per Move": 0.05,
        "Rook Lifts per Move": 0.02,
        "Central Pawn Breaks per Move": 0.05,
        "F7/F2 Attacks per Move": 0.05,
    }

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

    if verbose:
        for i, (weight, name, value) in enumerate(features):
            contribution = (weighted_scores[i] / total_weighted_score * 100) if total_weighted_score > 0 else 0
            cap = normalization_caps.get(name, 1.0)
            normalized_value = min(value / cap, 1.0) if cap > 0 else 0
            print(f"{name:<30} | {value:10.4f} | {normalized_value:10.4f} | {weighted_scores[i]:10.4f} | {contribution:9.2f}%")

    # Final score is the sum of weighted scores, scaled by total weight to a [0, 1] range
    final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    if verbose:
        print("-" * 78)

    return final_score


def main():
    parser = argparse.ArgumentParser(description="Chess Aggression Analysis Tool")
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
            
            # Use read_headers to quickly check player name before parsing the whole game
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                break # End of file

            if args.player in (headers.get("White", "?"), headers.get("Black", "?")):
                # Found a game, now create the game object from the headers
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