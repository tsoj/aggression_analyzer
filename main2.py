import time
import chess
import chess.pgn
import argparse
from dataclasses import dataclass, field
from typing import List

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

# --- Configuration ---
# A material advantage greater than this value means a player is already
# considered to be in a "winning" position. Sacrifices are not counted
# when the player is this far ahead.
WINNING_MATERIAL_ADVANTAGE = PIECE_VALUES[chess.PAWN] * 4

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
    forfeited_castling_games: int = 0
    pawn_storms_vs_king: int = 0
    total_checks: int = 0

    # 2. Pawn Structure & Line Opening
    central_pawn_breaks: int = 0

    # 3. Piece Activity & Placement
    advanced_pieces: int = 0
    rook_lifts: int = 0
    knight_outposts: int = 0
    # NEW: Proximity to King
    moves_near_king_dist: List[int] = field(default_factory=lambda: [0] * 8)
    captures_near_king_dist: List[int] = field(default_factory=lambda: [0] * 8)
    # NEW: Threats on King
    rook_queen_threats: int = 0
    bishop_queen_threats: int = 0


    # 4. Move Characteristics & Material Balance
    # NEW: Advanced sacrifice score (duration * depth)
    total_sacrifice_score: float = 0.0
    forcing_moves: int = 0 # Captures + Checks

    # 5. Tactical Patterns
    f7_f2_attacks: int = 0
    coordinated_attacks: int = 0


def get_material_score(board: chess.Board, side: chess.Color) -> int:
    """Calculates the material value for a given side using standard values."""
    score = 0
    # Using python-chess's built-in values (P=100, N=320, B=330, R=510, Q=900)
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        value = PIECE_VALUES[piece_type]
        score += len(board.pieces(piece_type, side)) * value
    return score / 100 # Return in pawn units

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

    # --- State for advanced sacrifice detection ---
    sacrifice_state = {'active': False, 'start_ply': 0, 'max_deficit': 0.0}

    for ply, move in enumerate(game.mainline_moves()):
        turn = board.turn

        # --- Pre-move analysis ---
        if turn == player_color:
            enemy_king_square = board.king(not player_color)
            dist_to_king = chess.square_distance(move.to_square, enemy_king_square)
            moving_piece_type = board.piece_type_at(move.from_square)

            # Proximity Analysis
            if board.is_capture(move):
                stats.captures_near_king_dist[dist_to_king] += 1
            else:
                stats.moves_near_king_dist[dist_to_king] += 1

            # Rook/Bishop/Queen Threats on King
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            king_file = chess.square_file(enemy_king_square)
            king_rank = chess.square_rank(enemy_king_square)

            if moving_piece_type in [chess.ROOK, chess.QUEEN]:
                if to_file == king_file or to_rank == king_rank:
                    stats.rook_queen_threats += 1
            if moving_piece_type in [chess.BISHOP, chess.QUEEN]:
                if abs(to_file - king_file) == abs(to_rank - king_rank):
                    stats.bishop_queen_threats += 1

        # --- Track opponent's castling ---
        else:
            if board.is_castling(move):
                them_castled_side = "K" if board.is_kingside_castling(move) else "Q"

        # --- Advanced Sacrifice Detection Logic (only for wins) ---
        if is_win:
            material_us = get_material_score(board, player_color)
            material_them = get_material_score(board, not player_color)
            balance = material_us - material_them

            if turn == player_color:
                if not sacrifice_state['active']:
                    # Check if a sacrifice is being initiated
                    if balance < WINNING_MATERIAL_ADVANTAGE:
                        board.push(move)
                        new_balance = get_material_score(board, player_color) - get_material_score(board, not player_color)
                        board.pop()
                        if new_balance < balance and new_balance < 0:
                            sacrifice_state['active'] = True
                            sacrifice_state['start_ply'] = ply
                            sacrifice_state['max_deficit'] = abs(new_balance)
                else: # Sacrifice is active
                    if balance >= 0: # Sacrifice ended, tally the score
                        duration = ply - sacrifice_state['start_ply']
                        stats.total_sacrifice_score += sacrifice_state['max_deficit'] * duration
                        sacrifice_state = {'active': False, 'start_ply': 0, 'max_deficit': 0.0} # Reset
                    else:
                        # Update the max deficit if it deepens
                        sacrifice_state['max_deficit'] = max(sacrifice_state['max_deficit'], abs(balance))

        # --- Standard Analysis (from previous script) ---
        if turn == player_color:


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
        if turn == player_color:
            stats.total_moves += 1

    # --- Post-game analysis ---
    if sacrifice_state['active']: # If game ends while sacrifice is active
        duration = ply - sacrifice_state['start_ply']
        stats.total_sacrifice_score += sacrifice_state['max_deficit'] * duration

    if not us_castled_side and ply >= 40:
        stats.forfeited_castling_games += 1

    stats.num_games += 1
    if is_win: stats.num_wins += 1
    elif game.headers["Result"] == "1/2-1/2": stats.num_draws += 1
    else: stats.num_losses += 1


def get_aggression_score(stats: AggressionStats, verbose: bool = False) -> float:
    if stats.num_games == 0: return 0.0

    # Helper function for proximity scores
    def get_proximity_score(distances: List[int], total_moves: int) -> float:
        if total_moves == 0: return 0.0
        weights = [0, 8, 6, 4, 2, 1, 0, 0] # High weight for moves/caps near king
        score = sum(weights[i] * freq for i, freq in enumerate(distances))
        max_score = sum(distances) * max(weights) if sum(distances) > 0 else 1
        return score / max_score

    features = [
        (20.0, "Sacrifice Score per Win", stats.total_sacrifice_score / stats.num_wins if stats.num_wins > 0 else 0),
        (12.0, "Captures Near King", get_proximity_score(stats.captures_near_king_dist, stats.forcing_moves)),
        (10.0, "Coordinated Attacks per Move", stats.coordinated_attacks / stats.total_moves if stats.total_moves > 0 else 0),
        (8.0, "Opposite-Side Castling Games", stats.opposite_side_castling_games / stats.num_games),
        (7.0, "Pawn Storms per Move", stats.pawn_storms_vs_king / stats.total_moves if stats.total_moves > 0 else 0),
        (6.0, "Rook/Queen Threats per Move", stats.rook_queen_threats / stats.total_moves if stats.total_moves > 0 else 0),
        (6.0, "Moves Near King", get_proximity_score(stats.moves_near_king_dist, stats.total_moves)),
        (6.0, "Advanced Pieces per Move", stats.advanced_pieces / stats.total_moves if stats.total_moves > 0 else 0),
        (5.0, "Forcing Moves per Move", stats.forcing_moves / stats.total_moves if stats.total_moves > 0 else 0),
        (5.0, "Checks per Move", stats.total_checks / stats.total_moves if stats.total_moves > 0 else 0),
        (5.0, "Forfeited Castling Games", stats.forfeited_castling_games / stats.num_games),
        (4.0, "Bishop/Queen Threats per Move", stats.bishop_queen_threats / stats.total_moves if stats.total_moves > 0 else 0),
        (4.0, "Knight Outposts per Move", stats.knight_outposts / stats.total_moves if stats.total_moves > 0 else 0),
        (3.0, "Rook Lifts per Move", stats.rook_lifts / stats.total_moves if stats.total_moves > 0 else 0),
        (3.0, "Central Pawn Breaks per Move", stats.central_pawn_breaks / stats.total_moves if stats.total_moves > 0 else 0),
        (2.0, "F7/F2 Attacks per Move", stats.f7_f2_attacks / stats.total_moves if stats.total_moves > 0 else 0),
    ]

    normalization_caps = {
        "Sacrifice Score per Win": 150.0, # A high score (e.g. Rook sac for 10 moves = 5*10*2=100)
        "Captures Near King": 1.0,
        "Coordinated Attacks per Move": 0.05,
        "Opposite-Side Castling Games": 0.5,
        "Pawn Storms per Move": 0.1,
        "Rook/Queen Threats per Move": 0.1,
        "Moves Near King": 1.0,
        "Forcing Moves per Move": 0.4,
        "Forfeited Castling Games": 0.3,
        "Bishop/Queen Threats per Move": 0.1,
    }

    total_weight = sum(w for w, n, v in features)
    total_weighted_score = 0
    verbose_data = []

    for weight, name, value in features:
        cap = normalization_caps.get(name, 1.0)
        normalized_value = min(value / cap, 1.0) if cap > 0 else 0
        weighted_score = weight * normalized_value
        total_weighted_score += weighted_score
        if verbose:
            verbose_data.append((name, value, normalized_value, weighted_score))

    final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    if verbose:
        print("\n--- Aggression Score Breakdown ---")
        print(f"{'Feature':<30} | {'Raw Value':>10} | {'Norm Value':>10} | {'Weighted':>10} | {'Contrib%':>10}")
        print("-" * 78)
        for name, value, normalized_value, weighted_score in verbose_data:
            contribution = (weighted_score / total_weighted_score * 100) if total_weighted_score > 0 else 0
            print(f"{name:<30} | {value:10.4f} | {normalized_value:10.4f} | {weighted_score:10.4f} | {contribution:9.2f}%")
        print("-" * 78)

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

            try:
                game = chess.pgn.read_game(pgn)
                if game is None: break
            except Exception as e:
                print(f"\nSkipping a game due to a parsing error: {e}")
                continue

            if args.player in (game.headers.get("White", "?"), game.headers.get("Black", "?")):
                analyse_game(game, args.player, stats)
                games_analyzed += 1
                print(f"\rAnalyzed {games_analyzed} games...", end="")

    print("\n\n--- Analysis Complete ---")
    print(f"Player: {args.player}")

    if stats.num_games > 0:
        print("\nOverall Stats:")
        print(f"- Games Analyzed: {stats.num_games}")
        print(f"- Wins:   {stats.num_wins} ({100*stats.num_wins/stats.num_games:.1f}%)")

        score = get_aggression_score(stats, args.verbose)

        print("\n--- Final Score ---")
        print(f"Aggression Score: {score:.4f}")
        print("(A score from 0.0 to 1.0 indicating aggressive tendencies, based on king attacks,")
        print(" and the depth and duration of material sacrifices in winning games.)")
    else:
        print("No games found for this player in the PGN file.")

if __name__ == '__main__':
    main()
