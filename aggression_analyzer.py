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
WINNING_MATERIAL_ADVANTAGE = PIECE_VALUES[chess.PAWN] * 3

# Feature weights
feature_weights = {
    "Advanced Pieces per Move": 0.402045,
    "Bishop/Queen Threats per Move": 0.783636,
    "Captures Near King": 0.676496,
    "Central Pawn Breaks per Move": 1.212612,
    "Checks per Move": 0.232617,
    "Coordinated Attacks per Move": 1.048515,
    "F7/F2 Attacks per Move": 0.674311,
    "Forcing Moves per Move": 0.134491,
    "Forfeited Castling Games": 1.818294,
    "Knight Outposts per Move": 1.075949,
    "Moves Near King": 1.045039,
    "Opposite-Side Castling Games": 1.842295,
    "Pawn Storms per Move": 1.250377,
    "Rook Lifts per Move": 1.999973,
    "Rook/Queen Threats per Move": 0.870153,
    "Sacrifice Score per Win Move": 1.931901,
    "Short Game Bonus per Win": 1.572457,
}

# Normalization parameters calculated from large dataset
normalization_params = {
  "Sacrifice Score per Win Move": {"mean": 0.0454765029435485, "std": 0.18451553383842167},
  "Captures Near King": {"mean": 0.2984049776112477, "std": 0.15439150647975677},
  "Coordinated Attacks per Move": {"mean": 0.026443699984891662, "std": 0.06003311001527992},
  "Opposite-Side Castling Games": {"mean": 0.05628226331879918, "std": 0.23047176024863106},
  "Pawn Storms per Move": {"mean": 0.09178490741602273, "std": 0.09173832576810186},
  "Rook/Queen Threats per Move": {"mean": 0.035637999800881684, "std": 0.04836623845195941},
  "Moves Near King": {"mean": 0.22080340429142714, "std": 0.22682390092521104},
  "Advanced Pieces per Move": {"mean": 0.14197955031688525, "std": 0.1070854156274747},
  "Forcing Moves per Move": {"mean": 0.23460229670308344, "std": 0.09857636693558647},
  "Checks per Move": {"mean": 0.044402369971270325, "std": 0.06128079312800122},
  "Forfeited Castling Games": {"mean": 0.10098732020247582, "std": 0.30131948447742696},
  "Bishop/Queen Threats per Move": {"mean": 0.03001069585649784, "std": 0.04570638515459593},
  "Knight Outposts per Move": {"mean": 0.012487240931485535, "std": 0.02620039632844376},
  "Rook Lifts per Move": {"mean": 0.008264354851863567, "std": 0.022507583275233375},
  "Central Pawn Breaks per Move": {"mean": 0.029584507031272526, "std": 0.040124917124545995},
  "Short Game Bonus per Win": {"mean": 0.1601129320570006, "std": 0.4057825759723069},
  "F7/F2 Attacks per Move": {"mean": 0.017473489556894473, "std": 0.031409328143718655}
}

# --- Data Classes ---
@dataclass
class AggressionStats:
    """Stores all statistics needed to calculate an aggression score for a player."""
    num_games: int = 0
    num_wins: int = 0
    num_draws: int = 0
    num_losses: int = 0
    total_moves: int = 0
    num_win_moves: int = 0
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
    short_game_bonus: float = 0.0

@dataclass
class MaterialBalance:
    """Tracks material balance and winning advantage status."""
    us: int
    them: int

    @property
    def balance(self) -> int:
        return self.us - self.them

    @property
    def has_winning_advantage(self) -> bool:
        return self.balance >= WINNING_MATERIAL_ADVANTAGE

@dataclass
class SacrificeState:
    """Tracks sacrifice sequences and deficits."""
    active: bool = False
    quiet_deficits: List[int] = field(default_factory=list)

# --- Core Analysis Functions ---
def get_material_score(board: chess.Board, side: chess.Color) -> int:
    """Calculates the material value for a given side using standard values."""
    score = 0
    for piece_type in PIECE_VALUES:
        score += len(board.pieces(piece_type, side)) * PIECE_VALUES[piece_type]
    return score

def get_material_balance(board: chess.Board, player_color: chess.Color) -> MaterialBalance:
    """Gets the current material balance for analysis."""
    material_us = get_material_score(board, player_color)
    material_them = get_material_score(board, not player_color)
    return MaterialBalance(material_us, material_them)

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

# --- Sacrifice Analysis ---
def _max_filter_radius(seq: List[int], radius: int) -> List[int]:
    """Return a list where each position is replaced by the max value in a window
    centered at that position with +/- radius bounds (within seq)."""
    n = len(seq)
    if n == 0:
        return []
    out = []
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out.append(max(seq[lo:hi]))
    return out

def _score_sacrifice_quiet_deficits(quiet_deficits: List[int], radius: int = 2) -> float:
    """Apply max filter and return the sum of filtered values (float)."""
    if not quiet_deficits:
        return 0.0
    filtered = _max_filter_radius(quiet_deficits, radius)
    return float(sum(filtered))

def update_sacrifice_tracking(board: chess.Board, move: chess.Move, player_color: chess.Color,
                            sacrifice_state: SacrificeState, is_win: bool, stats: AggressionStats):
    """Updates sacrifice tracking based on material balance after move."""
    # Temporarily push the move to check the material balance *after* our move.
    board.push(move)
    balance_after = get_material_balance(board, player_color)
    board.pop()  # Revert board state

    if balance_after.balance < 0:
        # We are in a deficit. Start or continue the deficit sequence.
        if not sacrifice_state.active:
            sacrifice_state.active = True
            sacrifice_state.quiet_deficits = []

        # If this move is QUIET (non-capture, non-check), record the deficit magnitude.
        if not board.is_capture(move) and not board.gives_check(move):
            sacrifice_state.quiet_deficits.append(abs(balance_after.balance))
    else:
        # No deficit; if a sequence was active, it ends here.
        if sacrifice_state.active:
            # If the game was won, score the completed sacrifice sequence.
            if is_win:
                seq_score = _score_sacrifice_quiet_deficits(sacrifice_state.quiet_deficits)
                stats.total_sacrifice_score += seq_score
            # Reset state.
            sacrifice_state.active = False
            sacrifice_state.quiet_deficits = []

def finalize_sacrifice_tracking(sacrifice_state: SacrificeState, is_win: bool, stats: AggressionStats):
    """Finalize sacrifice tracking at game end."""
    if sacrifice_state.active and is_win:
        seq_score = _score_sacrifice_quiet_deficits(sacrifice_state.quiet_deficits)
        stats.total_sacrifice_score += seq_score

# --- Move Analysis Functions ---
def analyze_castling(board: chess.Board, move: chess.Move, is_our_turn: bool,
                    us_castled_side: str, them_castled_side: str,
                    material_balance: MaterialBalance, stats: AggressionStats) -> Tuple[str, str]:
    """Analyzes castling moves and updates stats."""
    if not board.is_castling(move):
        return us_castled_side, them_castled_side

    side = "K" if board.is_kingside_castling(move) else "Q"

    if is_our_turn:
        us_castled_side = side
        if them_castled_side and us_castled_side != them_castled_side:
            # Only count as aggressive if we don't have winning material advantage
            if not material_balance.has_winning_advantage:
                stats.opposite_side_castling_games += 1
    else:
        them_castled_side = side

    return us_castled_side, them_castled_side

def analyze_king_proximity(analysis_board: chess.Board, analysis_move: chess.Move,
                          is_capture: bool, material_balance: MaterialBalance, stats: AggressionStats):
    """Analyzes moves near the enemy king."""


    enemy_king_square = analysis_board.king(chess.BLACK)
    if not enemy_king_square:
        return

    dist_to_king = chess.square_distance(analysis_move.to_square, enemy_king_square)

    if is_capture:
        stats.captures_near_king_dist[dist_to_king] += 1
    else:
        stats.moves_near_king_dist[dist_to_king] += 1

def analyze_piece_threats(analysis_board: chess.Board, analysis_move: chess.Move,
                         moving_piece_type: int, material_balance: MaterialBalance, stats: AggressionStats):
    """Analyzes piece threats to the enemy king."""


    enemy_king_square = analysis_board.king(chess.BLACK)
    if not enemy_king_square:
        return

    to_file, to_rank = chess.square_file(analysis_move.to_square), chess.square_rank(analysis_move.to_square)
    king_file, king_rank = chess.square_file(enemy_king_square), chess.square_rank(enemy_king_square)

    if moving_piece_type in [chess.ROOK, chess.QUEEN] and (to_file == king_file or to_rank == king_rank):
        stats.rook_queen_threats += 1

    if moving_piece_type in [chess.BISHOP, chess.QUEEN] and abs(to_file - king_file) == abs(to_rank - king_rank):
        stats.bishop_queen_threats += 1

def analyze_tactical_moves(board: chess.Board, analysis_board: chess.Board, analysis_move: chess.Move,
                          move: chess.Move, moving_piece_type: int, them_castled_side: str,
                          player_color: chess.Color, material_balance: MaterialBalance, stats: AggressionStats):
    """Analyzes various tactical move types."""


    # Pawn storms
    if moving_piece_type == chess.PAWN and them_castled_side:
        pawn_file = chess.square_file(move.from_square)
        original_king_sq = board.king(not player_color)
        if original_king_sq:
            king_file = chess.square_file(original_king_sq)
            if abs(pawn_file - king_file) <= 2:
                stats.pawn_storms_vs_king += 1

    # Central pawn breaks
    if (moving_piece_type == chess.PAWN and
        chess.square_file(analysis_move.from_square) in [3, 4] and
        chess.square_rank(analysis_move.to_square) == 4):
        stats.central_pawn_breaks += 1

    # Advanced pieces
    if moving_piece_type != chess.PAWN and chess.square_rank(analysis_move.to_square) >= 4:
        stats.advanced_pieces += 1

    # Rook lifts
    if (moving_piece_type == chess.ROOK and
        chess.square_rank(analysis_move.from_square) in [0, 1] and
        chess.square_rank(analysis_move.to_square) == 2):
        stats.rook_lifts += 1

    # Knight outposts
    if moving_piece_type == chess.KNIGHT and chess.square_rank(analysis_move.to_square) >= 4:
        if any(analysis_board.piece_type_at(sq) == chess.PAWN
               for sq in analysis_board.attackers(chess.WHITE, analysis_move.to_square)):
            stats.knight_outposts += 1

    # F7/F2 attacks
    if (analysis_move.to_square == chess.F7 or
        chess.F7 in analysis_board.attacks(analysis_move.to_square)):
        stats.f7_f2_attacks += 1

def analyze_forcing_moves(board: chess.Board, move: chess.Move, material_balance: MaterialBalance, stats: AggressionStats):
    """Analyzes forcing moves (captures and checks)."""


    if board.is_capture(move) or board.gives_check(move):
        stats.forcing_moves += 1

    if board.gives_check(move):
        stats.total_checks += 1

def analyze_coordinated_attacks(board: chess.Board, player_color: chess.Color,
                               material_balance: MaterialBalance, stats: AggressionStats):
    """Analyzes coordinated attacks around the enemy king."""


    enemy_king_square = board.king(not player_color)
    if not enemy_king_square:
        return

    # Check 3x3 area around the king (all squares within distance 1)
    unique_attackers = set()
    for square in chess.SQUARES:
        if chess.square_distance(square, enemy_king_square) <= 1:
            attackers = board.attackers(player_color, square)
            unique_attackers.update(attackers)

    if len(unique_attackers) >= 3:
        stats.coordinated_attacks += 1

def calculate_short_game_bonus(board: chess.Board, player_color: chess.Color, ply: int) -> float:
    """Calculates bonus for winning short games without material advantage."""
    final_balance = get_material_balance(board, player_color)

    # Only award short game bonus if the win wasn't achieved through overwhelming material advantage
    if final_balance.has_winning_advantage:
        return 0.0

    game_length = (ply + 1) // 2  # Convert half-moves to full moves
    if game_length <= 60:
        return max(0, (60 - game_length) / 30.0)  # Linear decrease from 1.0 to 0
    return 0.0

# --- Main Analysis Function ---
def analyse_game(game: chess.pgn.Game, player_name: str, stats: AggressionStats):
    """
    Analyzes a single game for the specified player and updates the stats object.
    """
    player_color = chess.WHITE if game.headers["White"] == player_name else chess.BLACK
    board = game.board()
    us_castled_side = ""
    them_castled_side = ""

    termination = game.headers.get("Termination", "").lower()
    is_draw = "time forfeit" in termination or game.headers["Result"] == "1/2-1/2"
    is_win = ((game.headers["Result"] == "1-0" and player_color == chess.WHITE) or
              (game.headers["Result"] == "0-1" and player_color == chess.BLACK)) and not is_draw

    sacrifice_state = SacrificeState()
    ply = 0

    for ply, move in enumerate(game.mainline_moves()):
        turn = board.turn
        is_our_turn = (turn == player_color)
        material_balance = get_material_balance(board, player_color)

        # Handle castling (needs to track sides regardless of material balance)
        us_castled_side, them_castled_side = analyze_castling(
            board, move, is_our_turn, us_castled_side or "", them_castled_side or "",
            material_balance, stats
        )

        if is_our_turn:
            analysis_board, analysis_move = create_analysis_view(board, move, player_color)
            moving_piece_type = analysis_board.piece_type_at(analysis_move.from_square)

            if not moving_piece_type:
                board.push(move)
                continue

            # Update sacrifice tracking
            update_sacrifice_tracking(board, move, player_color, sacrifice_state, is_win, stats)

            # Only analyze aggression-related moves if we don't have a winning material advantage
            if not material_balance.has_winning_advantage:
                # Analyze various aspects of the move
                analyze_king_proximity(analysis_board, analysis_move,
                                     analysis_board.is_capture(analysis_move), material_balance, stats)

                analyze_piece_threats(analysis_board, analysis_move, moving_piece_type,
                                    material_balance, stats)

                analyze_tactical_moves(board, analysis_board, analysis_move, move,
                                     moving_piece_type, them_castled_side, player_color,
                                     material_balance, stats)

                analyze_forcing_moves(board, move, material_balance, stats)

                # Analyze coordinated attacks
                analyze_coordinated_attacks(board, player_color, material_balance, stats)

        board.push(move)

        if is_our_turn:
            stats.total_moves += 1
            if is_win:
                stats.num_win_moves += 1

    # Finalize sacrifice tracking
    finalize_sacrifice_tracking(sacrifice_state, is_win, stats)

    # Check for forfeited castling
    if (not us_castled_side or us_castled_side == "") and ply >= 40:
        stats.forfeited_castling_games += 1

    # Update game results
    stats.num_games += 1
    if is_draw:
        stats.num_draws += 1
    elif is_win:
        stats.num_wins += 1
        stats.short_game_bonus += calculate_short_game_bonus(board, player_color, ply)
    else:
        stats.num_losses += 1

# --- Score Calculation Functions ---
def get_raw_feature_scores(stats: AggressionStats) -> Dict[str, float]:
    """Calculates raw feature scores without weighting or normalization."""
    if stats.num_games == 0 or stats.total_moves == 0:
        return {}

    def get_proximity_score(distances: List[int]) -> float:
        weights = [0, 8, 6, 4, 2, 1, 0, 0]
        score = sum(weights[i] * freq for i, freq in enumerate(distances))
        total_moves_in_zone = sum(distances)
        return score / (total_moves_in_zone * max(weights)) if total_moves_in_zone > 0 else 0.0

    raw_scores = {
        "Sacrifice Score per Win Move": stats.total_sacrifice_score / stats.num_win_moves if stats.num_win_moves > 0 else 0,
        "Captures Near King": get_proximity_score(stats.captures_near_king_dist),
        "Coordinated Attacks per Move": stats.coordinated_attacks / stats.total_moves,
        "Opposite-Side Castling Games": stats.opposite_side_castling_games / stats.num_games,
        "Pawn Storms per Move": stats.pawn_storms_vs_king / stats.total_moves,
        "Rook/Queen Threats per Move": stats.rook_queen_threats / stats.total_moves,
        "Moves Near King": get_proximity_score(stats.moves_near_king_dist),
        "Advanced Pieces per Move": stats.advanced_pieces / stats.total_moves,
        "Forcing Moves per Move": stats.forcing_moves / stats.total_moves,
        "Checks per Move": stats.total_checks / stats.total_moves,
        "Forfeited Castling Games": stats.forfeited_castling_games / stats.num_games,
        "Bishop/Queen Threats per Move": stats.bishop_queen_threats / stats.total_moves,
        "Knight Outposts per Move": stats.knight_outposts / stats.total_moves,
        "Rook Lifts per Move": stats.rook_lifts / stats.total_moves,
        "Central Pawn Breaks per Move": stats.central_pawn_breaks / stats.total_moves,
        "Short Game Bonus per Win": stats.short_game_bonus / stats.num_wins if stats.num_wins > 0 else 0,
        "F7/F2 Attacks per Move": stats.f7_f2_attacks / stats.total_moves,
    }
    return raw_scores

def get_aggression_score(stats: AggressionStats, verbose: bool = False) -> float:
    """Calculates the final aggression score from a stats object using weights and normalization."""
    raw_scores = get_raw_feature_scores(stats)
    if not raw_scores:
        return 0.0

    total_weight = sum(feature_weights.values())
    total_weighted_score = 0

    for feature_name, raw_value in raw_scores.items():
        weight = feature_weights[feature_name]
        if feature_name in normalization_params:
            params = normalization_params[feature_name]
            if params['std'] > 0:
                # Normalize using statistical parameters: (value - mean) / std
                normalized_value = (raw_value - params['mean']) / params['std']
            else:
                normalized_value = 0.0
        else:
            # Fallback for unknown features
            normalized_value = raw_value
        total_weighted_score += weight * normalized_value

    final_score = (total_weighted_score / total_weight)
    return final_score

# --- Utility Functions ---
def download_and_decompress_pgn(pgn_url: str, output_path: str) -> bool:
    """Downloads and decompresses a PGN file from a URL."""
    print("Downloading and decompressing PGN...")
    try:
        with requests.get(pgn_url, stream=True) as response:
            response.raise_for_status()
            decompressor = zstd.ZstdDecompressor()
            with open(output_path, 'wb') as out_file:
                # Read compressed data and decompress it
                reader = decompressor.stream_reader(response.content)
                out_file.write(reader.read())
        print(f"Successfully saved to {output_path}")
        return True
    except (requests.exceptions.RequestException, zstd.ZstdError, IOError) as e:
        print(f"Error during download/decompression: {e}")
        return False

def should_include_game(game: chess.pgn.Game, args) -> bool:
    """Determines if a game should be included based on filters."""
    # Apply event filter if specified
    if args.event_filter:
        event = game.headers.get("Event", "").lower()
        if not any(filter_type.lower() in event for filter_type in args.event_filter):
            return False

    # Check player names
    white_player = game.headers.get("White", "?")
    black_player = game.headers.get("Black", "?")
    if "?" in (white_player, black_player):
        return False

    # Filter by minimum rating
    if args.min_rating:
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            min_elo = min(white_elo, black_elo)
            if min_elo < args.min_rating:
                return False
        except (ValueError, TypeError):
            return False

    return True

def process_single_player_mode(args, pgn_path: str) -> None:
    """Processes games in single player mode."""
    game_scores_for_player = []
    games_processed = 0
    games_filtered_by_rating = 0

    if args.min_rating:
        print(f"Analyzing games for player '{args.player}' (filtering games where lower-rated player is below {args.min_rating})...")
    else:
        print(f"Analyzing games for player '{args.player}'...")

    with open(pgn_path) as pgn_file:
        while True:
            if args.games and games_processed >= args.games:
                print(f"\nReached game limit of {args.games}.")
                break

            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
            except Exception:
                continue

            if not should_include_game(game, args):
                if args.min_rating:
                    games_filtered_by_rating += 1
                continue

            white_player = game.headers.get("White", "?")
            black_player = game.headers.get("Black", "?")

            if args.player in (white_player, black_player):
                stats = AggressionStats()
                analyse_game(game, args.player, stats)
                score = get_aggression_score(stats)
                game_scores_for_player.append((game, score))

            games_processed += 1
            print(f"\rProcessed {games_processed} games...", end="")

    print(f"\n\n--- Analysis Complete ---")
    if games_filtered_by_rating > 0 and args.min_rating:
        print(f"Filtered out {games_filtered_by_rating} games due to rating requirements (min rating: {args.min_rating})")

    # Output results
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

def process_all_players_mode(args, pgn_path: str) -> None:
    """Processes games in all players mode."""
    all_player_stats: Dict[str, AggressionStats] = {}
    top_aggressive_games = []
    least_aggressive_games = []
    games_processed = 0
    games_filtered_by_rating = 0

    if args.min_rating:
        print(f"Analyzing all players (filtering games where lower-rated player is below {args.min_rating})...")
    else:
        print("Analyzing all players...")
    with open(pgn_path) as pgn_file:
        while True:
            if args.games and games_processed >= args.games:
                print(f"\nReached game limit of {args.games}.")
                break

            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
            except Exception:
                continue

            if not should_include_game(game, args):
                if args.min_rating:
                    games_filtered_by_rating += 1
                continue

            white_player = game.headers.get("White", "?")
            black_player = game.headers.get("Black", "?")

            # Analyze for both players
            stats_white = all_player_stats.setdefault(white_player, AggressionStats())
            analyse_game(game, white_player, stats_white)
            stats_black = all_player_stats.setdefault(black_player, AggressionStats())
            analyse_game(game, black_player, stats_black)

            # Track top/least aggressive games across all players
            for player in [white_player, black_player]:
                temp_stats = AggressionStats()
                analyse_game(game, player, temp_stats)
                score = get_aggression_score(temp_stats)

                if len(top_aggressive_games) < args.top_n or score > top_aggressive_games[-1][1]:
                    top_aggressive_games.append((game, score, player, temp_stats))
                    top_aggressive_games.sort(key=lambda item: item[1], reverse=True)
                    if len(top_aggressive_games) > args.top_n:
                        top_aggressive_games.pop()

                if len(least_aggressive_games) < args.top_n or score < least_aggressive_games[-1][1]:
                    least_aggressive_games.append((game, score, player, temp_stats))
                    least_aggressive_games.sort(key=lambda item: item[1])
                    if len(least_aggressive_games) > args.top_n:
                        least_aggressive_games.pop()

            games_processed += 1
            print(f"\rProcessed {games_processed} games...", end="")

    print(f"\n\n--- Analysis Complete ---")
    if games_filtered_by_rating > 0:
        if args.min_rating:
            print(f"Filtered out {games_filtered_by_rating} games due to rating requirements (min rating: {args.min_rating})")

    # Output results for all players mode
    player_results = []
    for player, stats in all_player_stats.items():
        if stats.num_games >= args.min_games:
            score = get_aggression_score(stats, args.verbose and stats.num_games > 0)
            player_results.append((player, score, stats))

    if not player_results:
        print(f"No players found with at least {args.min_games} games.")
    else:
        sorted_players = sorted(player_results, key=lambda item: item[1], reverse=True)

        print(f"Aggression ranking for {len(sorted_players)} players with at least {args.min_games} games:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Player':<30} {'Agg. Score':<15} {'Games':<10} {'Record (W/D/L)'}")
        print("-" * 80)
        for i, (player, score, stats) in enumerate(sorted_players):
            record = f"{stats.num_wins} / {stats.num_draws} / {stats.num_losses}"
            print(f"{i+1:<5} {player:<30} {score:<15.2f} {stats.num_games:<10} {record}")

    print(f"\n--- Top {args.top_n} Most Aggressive Games (All Players) ---")
    for game, score, player, stats in top_aggressive_games:
        print("-"*50)
        print(f"\nScore: {score:.2f} - {game.headers.get('Site', '?')} - {player}")
        print(f"White: {game.headers.get('White', '?')}, Black: {game.headers.get('Black', '?')}")
        print(game)
        print(stats)

    print(f"\n--- Top {args.top_n} Least Aggressive Games (All Players) ---")
    for game, score, player, stats in least_aggressive_games:
        print("-"*50)
        print(f"\nScore: {score:.2f} - {game.headers.get('Site', '?')} - {player}")
        print(f"White: {game.headers.get('White', '?')}, Black: {game.headers.get('Black', '?')}")
        print(game)
        print(stats)

def main():
    """Main entry point for the chess aggression analysis tool."""
    parser = argparse.ArgumentParser(description="Advanced Chess Aggression Analysis Tool.")
    parser.add_argument("--pgn", type=str, required=True, help="Path or URL to the PGN file (can be .pgn.zst).")
    parser.add_argument("--player", type=str, help="Name of a specific player to analyze. If omitted, all players will be analyzed.")
    parser.add_argument("--games", type=int, help="Maximum number of games to process.")
    parser.add_argument("--min_games", type=int, default=10, help="Minimum games for a player to be included in 'all players' analysis.")
    parser.add_argument("--min_rating", type=int, default=2000, help="Minimum rating for the lower-rated player in each game.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top/bottom games to display.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown of the aggression score.")
    parser.add_argument("--event_filter", type=str, nargs="+", help="Filter games by event types (e.g., 'rapid', 'classical'). If omitted, all events are included.")
    args = parser.parse_args()

    # Handle PGN download if URL is provided
    pgn_path = args.pgn
    tmpdir = None
    if pgn_path.startswith("http"):
        tmpdir = tempfile.TemporaryDirectory()
        local_pgn_path = os.path.join(tmpdir.name, "games.pgn")
        if not download_and_decompress_pgn(pgn_path, local_pgn_path):
            return
        pgn_path = local_pgn_path

    try:
        # Process games based on mode
        if args.player:
            process_single_player_mode(args, pgn_path)
        else:
            process_all_players_mode(args, pgn_path)
    finally:
        # Cleanup temporary directory if it was created
        if tmpdir is not None:
            tmpdir.cleanup()

if __name__ == '__main__':
    main()
