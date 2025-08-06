import time
import chess
import chess.pgn
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import requests
import zstandard as zstd
import os
import tempfile

# --- Constants ---
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}
WINNING_MATERIAL_ADVANTAGE = PIECE_VALUES[chess.PAWN] * 4

# --- Data Class for Statistics ---
@dataclass
class AggressionStats:
    """Stores all statistics needed to calculate an aggression score for a player."""
    num_games: int = 0
    num_wins: int = 0
    num_draws: int = 0
    num_losses: int = 0
    total_moves: int = 0
    opposite_side_castling_games: int = 0
    forfeited_castling_games: int = 0
    pawn_storms_vs_king: int = 0
    total_checks: int = 0
    central_pawn_breaks: int = 0
    advanced_pieces: int = 0
    rook_lifts: int = 0
    knight_outposts: int = 0
    moves_near_king_dist: List[int] = field(default_factory=lambda: [0] * 8)
    captures_near_king_dist: List[int] = field(default_factory=lambda: [0] * 8)
    rook_queen_threats: int = 0
    bishop_queen_threats: int = 0
    total_sacrifice_score: float = 0.0
    forcing_moves: int = 0
    f7_f2_attacks: int = 0
    coordinated_attacks: int = 0

# --- Core Analysis Functions (Unchanged) ---
def get_material_score(board: chess.Board, side: chess.Color) -> int:
    """Calculates the material value for a given side using standard values."""
    score = 0
    for piece_type in PIECE_VALUES:
        score += len(board.pieces(piece_type, side)) * PIECE_VALUES[piece_type]
    return score

def create_analysis_view(board: chess.Board, move: chess.Move, player_color: chess.Color) -> Tuple[chess.Board, chess.Move]:
    """Creates a board view where the player to be analyzed is always White."""
    if player_color == chess.WHITE:
        return board, move
    analysis_board = board.copy(stack=False)
    analysis_board.mirror()
    analysis_move = chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion
    )
    return analysis_board, analysis_move

def analyse_game(game: chess.pgn.Game, player_name: str, stats: AggressionStats):
    """
    Analyzes a single game for the specified player and updates the stats object.
    This function's internal logic remains the same.
    """
    player_color = chess.WHITE if game.headers["White"] == player_name else chess.BLACK
    board = game.board()
    us_castled_side = None
    them_castled_side = None

    is_win = (game.headers["Result"] == "1-0" and player_color == chess.WHITE) or \
             (game.headers["Result"] == "0-1" and player_color == chess.BLACK)

    sacrifice_state = {'active': False, 'start_ply': 0, 'max_deficit': 0.0}
    ply = 0 # Initialize ply
    for ply, move in enumerate(game.mainline_moves()):
        turn = board.turn
        is_our_turn = (turn == player_color)

        if board.is_castling(move):
            side = "K" if board.is_kingside_castling(move) else "Q"
            if is_our_turn:
                us_castled_side = side
                if them_castled_side and us_castled_side != them_castled_side:
                    stats.opposite_side_castling_games += 1
            else:
                them_castled_side = side

        if is_our_turn:
            analysis_board, analysis_move = create_analysis_view(board, move, player_color)
            moving_piece_type = analysis_board.piece_type_at(analysis_move.from_square)
            if not moving_piece_type:
                 board.push(move)
                 continue

            enemy_king_square = analysis_board.king(chess.BLACK)
            dist_to_king = chess.square_distance(analysis_move.to_square, enemy_king_square)
            if analysis_board.is_capture(analysis_move):
                stats.captures_near_king_dist[dist_to_king] += 1
            else:
                stats.moves_near_king_dist[dist_to_king] += 1

            to_file, to_rank = chess.square_file(analysis_move.to_square), chess.square_rank(analysis_move.to_square)
            king_file, king_rank = chess.square_file(enemy_king_square), chess.square_rank(enemy_king_square)

            if moving_piece_type in [chess.ROOK, chess.QUEEN] and (to_file == king_file or to_rank == king_rank):
                stats.rook_queen_threats += 1
            if moving_piece_type in [chess.BISHOP, chess.QUEEN] and abs(to_file - king_file) == abs(to_rank - king_rank):
                stats.bishop_queen_threats += 1

            if moving_piece_type == chess.PAWN and them_castled_side:
                pawn_file = chess.square_file(move.from_square)
                original_king_sq = board.king(not player_color)
                king_file = chess.square_file(original_king_sq)
                if abs(pawn_file - king_file) <= 2:
                    stats.pawn_storms_vs_king += 1

            if moving_piece_type == chess.PAWN and chess.square_file(analysis_move.from_square) in [3, 4] and chess.square_rank(analysis_move.to_square) == 4:
                stats.central_pawn_breaks += 1

            if moving_piece_type != chess.PAWN and chess.square_rank(analysis_move.to_square) >= 4:
                stats.advanced_pieces += 1

            if moving_piece_type == chess.ROOK and chess.square_rank(analysis_move.from_square) in [0, 1] and chess.square_rank(analysis_move.to_square) == 2:
                stats.rook_lifts += 1

            if moving_piece_type == chess.KNIGHT and chess.square_rank(analysis_move.to_square) >= 4:
                if any(analysis_board.piece_type_at(sq) == chess.PAWN for sq in analysis_board.attackers(chess.WHITE, analysis_move.to_square)):
                    stats.knight_outposts += 1

            if analysis_move.to_square == chess.F7 or chess.F7 in analysis_board.attacks(analysis_move.to_square):
                stats.f7_f2_attacks += 1

        if is_win and is_our_turn:
            material_us = get_material_score(board, player_color)
            material_them = get_material_score(board, not player_color)
            balance = material_us - material_them
            if not sacrifice_state['active']:
                if balance < (WINNING_MATERIAL_ADVANTAGE * -1):
                    board.push(move)
                    new_balance = get_material_score(board, player_color) - get_material_score(board, not player_color)
                    board.pop()
                    if new_balance < balance and new_balance < 0:
                        sacrifice_state.update({'active': True, 'start_ply': ply, 'max_deficit': abs(new_balance) / 100.0})
            else:
                if balance >= 0:
                    duration = ply - sacrifice_state['start_ply']
                    stats.total_sacrifice_score += sacrifice_state['max_deficit'] * duration
                    sacrifice_state.update({'active': False, 'start_ply': 0, 'max_deficit': 0.0})
                else:
                    sacrifice_state['max_deficit'] = max(sacrifice_state['max_deficit'], abs(balance) / 100.0)

        if is_our_turn:
            if board.is_capture(move) or board.gives_check(move):
                stats.forcing_moves += 1
            if board.gives_check(move):
                stats.total_checks += 1

        enemy_king_square = board.king(not player_color)
        if enemy_king_square and len(board.attackers(player_color, enemy_king_square)) >= 3:
            stats.coordinated_attacks += 1

        board.push(move)
        if is_our_turn:
            stats.total_moves += 1

    if sacrifice_state['active']:
        duration = (ply + 1) - sacrifice_state['start_ply']
        stats.total_sacrifice_score += sacrifice_state['max_deficit'] * duration

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
    """Calculates the final aggression score from a stats object. Unchanged."""
    if stats.num_games == 0 or stats.total_moves == 0: return 0.0

    def get_proximity_score(distances: List[int]) -> float:
        weights = [0, 8, 6, 4, 2, 1, 0, 0]
        score = sum(weights[i] * freq for i, freq in enumerate(distances))
        total_moves_in_zone = sum(distances)
        return score / (total_moves_in_zone * max(weights)) if total_moves_in_zone > 0 else 0.0

    features = [
        (20.0, "Sacrifice Score per Win", stats.total_sacrifice_score / stats.num_wins if stats.num_wins > 0 else 0),
        (12.0, "Captures Near King", get_proximity_score(stats.captures_near_king_dist)),
        (10.0, "Coordinated Attacks per Move", stats.coordinated_attacks / stats.total_moves),
        (8.0, "Opposite-Side Castling Games", stats.opposite_side_castling_games / stats.num_games),
        (7.0, "Pawn Storms per Move", stats.pawn_storms_vs_king / stats.total_moves),
        (6.0, "Rook/Queen Threats per Move", stats.rook_queen_threats / stats.total_moves),
        (6.0, "Moves Near King", get_proximity_score(stats.moves_near_king_dist)),
        (6.0, "Advanced Pieces per Move", stats.advanced_pieces / stats.total_moves),
        (5.0, "Forcing Moves per Move", stats.forcing_moves / stats.total_moves),
        (5.0, "Checks per Move", stats.total_checks / stats.total_moves),
        (5.0, "Forfeited Castling Games", stats.forfeited_castling_games / stats.num_games),
        (4.0, "Bishop/Queen Threats per Move", stats.bishop_queen_threats / stats.total_moves),
        (4.0, "Knight Outposts per Move", stats.knight_outposts / stats.total_moves),
        (3.0, "Rook Lifts per Move", stats.rook_lifts / stats.total_moves),
        (3.0, "Central Pawn Breaks per Move", stats.central_pawn_breaks / stats.total_moves),
        (2.0, "F7/F2 Attacks per Move", stats.f7_f2_attacks / stats.total_moves),
    ]
    normalization_caps = { "Sacrifice Score per Win": 50.0, "Captures Near King": 1.0, "Coordinated Attacks per Move": 0.05, "Opposite-Side Castling Games": 0.5, "Pawn Storms per Move": 0.1, "Rook/Queen Threats per Move": 0.1, "Moves Near King": 1.0, "Forcing Moves per Move": 0.4, "Forfeited Castling Games": 0.3, "Bishop/Queen Threats per Move": 0.1}

    total_weight, total_weighted_score = sum(w for w, _, _ in features), 0
    for weight, name, value in features:
        cap = normalization_caps.get(name, 0.2)
        normalized_value = min(value / cap, 1.0) if cap > 0 else 0
        total_weighted_score += weight * normalized_value

    final_score = (total_weighted_score / total_weight) * 100
    return final_score

# --- Utility and Main ---
def download_and_decompress_pgn(pgn_url: str, output_path: str):
    """Downloads and decompresses a PGN file from a URL."""
    print("Downloading and decompressing PGN...")
    try:
        with requests.get(pgn_url, stream=True) as response:
            response.raise_for_status()
            decompressor = zstd.ZstdDecompressor()
            with open(output_path, 'wb') as out_file, response.raw as in_stream:
                decompressor.copy_stream(in_stream, out_file)
        print(f"Successfully saved to {output_path}")
        return True
    except (requests.exceptions.RequestException, zstd.ZstdError, IOError) as e:
        print(f"Error during download/decompression: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Advanced Chess Aggression Analysis Tool.")
    parser.add_argument("--pgn", type=str, required=True, help="Path or URL to the PGN file (can be .pgn.zst).")
    parser.add_argument("--player", type=str, help="Name of a specific player to analyze. If omitted, all players will be analyzed.")
    parser.add_argument("--games", type=int, help="Maximum number of games to process.")
    parser.add_argument("--min_games", type=int, default=10, help="Minimum games for a player to be included in 'all players' analysis.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top/bottom games to display in single-player mode.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown of the aggression score.")
    args = parser.parse_args()

    pgn_path = args.pgn
    if pgn_path.startswith("http"):
        tmpdir = tempfile.TemporaryDirectory()
        local_pgn_path = os.path.join(tmpdir.name, "games.pgn")
        if not download_and_decompress_pgn(pgn_path, local_pgn_path):
            return
        pgn_path = local_pgn_path

    # --- Analysis ---
    all_player_stats: Dict[str, AggressionStats] = {}
    game_scores_for_player = [] # Used only in single-player mode
    games_processed = 0

    print("Analyzing games...")
    with open(pgn_path) as pgn_file:
        while True:
            if args.games and games_processed >= args.games:
                print(f"\nReached game limit of {args.games}.")
                break
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None: break
            except Exception as e:
                # This can happen with malformed PGNs
                continue

            games_processed += 1
            print(f"\rProcessed {games_processed} games...", end="")

            white_player = game.headers.get("White", "?")
            black_player = game.headers.get("Black", "?")
            if "?" in (white_player, black_player): continue

            # --- Mode Dispatch ---
            if args.player: # SINGLE-PLAYER MODE
                if args.player in (white_player, black_player):
                    stats = AggressionStats()
                    analyse_game(game, args.player, stats)
                    score = get_aggression_score(stats)
                    game_scores_for_player.append((game, score))
            else: # ALL-PLAYERS MODE
                stats_white = all_player_stats.setdefault(white_player, AggressionStats())
                analyse_game(game, white_player, stats_white)
                stats_black = all_player_stats.setdefault(black_player, AggressionStats())
                analyse_game(game, black_player, stats_black)

    print("\n\n--- Analysis Complete ---")

    # --- Output Results ---
    if args.player: # SINGLE-PLAYER MODE OUTPUT
        if not game_scores_for_player:
            print(f"No games found for player '{args.player}'.")
            return

        print(f"Player: {args.player}")
        total_games = len(game_scores_for_player)
        avg_score = sum(s for _, s in game_scores_for_player) / total_games

        print(f"\nOverall Stats for {total_games} games:")
        print(f"Average Aggression Score: {avg_score:.2f} / 100.0")

        sorted_games = sorted(game_scores_for_player, key=lambda item: item[1])

        print(f"\n--- Top {args.top_n} Most Aggressive Games ---")
        for game, score in reversed(sorted_games[-args.top_n:]):
            print(f"\nScore: {score:.2f} - {game.headers.get('Site', '?')}")
            print(game)

        print(f"\n--- Top {args.top_n} Least Aggressive Games ---")
        for game, score in sorted_games[:args.top_n]:
            print(f"\nScore: {score:.2f} - {game.headers.get('Site', '?')}")
            print(game)

    else: # ALL-PLAYERS MODE OUTPUT
        player_results = []
        for player, stats in all_player_stats.items():
            if stats.num_games >= args.min_games:
                score = get_aggression_score(stats, args.verbose and stats.num_games > 0)
                player_results.append((player, score, stats))

        if not player_results:
            print(f"No players found with at least {args.min_games} games.")
            return

        sorted_players = sorted(player_results, key=lambda item: item[1], reverse=True)

        print(f"Aggression ranking for {len(sorted_players)} players with at least {args.min_games} games:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Player':<30} {'Agg. Score':<15} {'Games':<10} {'Record (W/D/L)'}")
        print("-" * 80)
        for i, (player, score, stats) in enumerate(sorted_players):
            record = f"{stats.num_wins} / {stats.num_draws} / {stats.num_losses}"
            print(f"{i+1:<5} {player:<30} {score:<15.2f} {stats.num_games:<10} {record}")

    # Cleanup temporary directory if it was created
    if 'tmpdir' in locals():
        tmpdir.cleanup()

if __name__ == '__main__':
    main()
