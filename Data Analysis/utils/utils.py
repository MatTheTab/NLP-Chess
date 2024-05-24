import chess.pgn
import zstandard
import io
import numpy as np
import pandas as pd
import chess.engine
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple


def save_game_data(all_games_df: pd.DataFrame,
                   game_number: int,
                   game: chess.pgn.Game,
                   columns: List[str],
                   engine: chess.engine.SimpleEngine,
                   time_limit: float = 0.01) -> pd.DataFrame:
    '''
    Save game data from a chess game into a DataFrame.

    Parameters:
    - all_games_df (DataFrame): DataFrame containing all games data.
    - game_number (int): The number of the game.
    - game (chess.pgn.Game): The chess game object.
    - columns (list): List of column names for the DataFrame.
    - engine (chess.engine.SimpleEngine): The chess engine used for analysis.
    - time_limit (float, optional): Time limit for the engine
    analysis in seconds. Defaults to 0.01.

    Returns:
    DataFrame: Updated DataFrame containing the saved game data.
    '''
    data = {}
    for col in columns:
        data[col] = []
    board = game.board()
    j = 0
    white_elo = game.headers['WhiteElo']
    black_elo = game.headers['BlackElo']
    for move in game.mainline_moves():
        j += 1
        color = board.turn
        data, board = get_random_move(data, board, game_number,
                                      j, engine, color, time_limit,
                                      white_elo, black_elo)
        data, board = get_human_move(data, board, move,
                                     game_number, j, engine, color,
                                     time_limit, white_elo, black_elo)
    data_df = pd.DataFrame(data)
    if all_games_df.empty:
        all_games_df = data_df.copy()
    else:
        all_games_df = pd.concat([all_games_df, data_df], ignore_index=True)
    return all_games_df


def save_data(pgn_file_path: str,
              save_file_path: str,
              max_num_games: int,
              stockfish_path: str,
              shuffle: bool = True,
              verbose: bool = True,
              seed: int = 42) -> None:
    '''
    Save data from multiple chess games into a compressed CSV file.

    Parameters:
    - pgn_file_path (str): Path to the PGN file containing the chess games.
    - save_file_path (str): Path to save the compressed CSV file.
    - max_num_games (int): Maximum number of games to process.
    - stockfish_path (str): Path to the Stockfish engine executable.
    - shuffle (bool, optional): Whether to shuffle the data.
      Defaults to True.
    - verbose (bool, optional): Whether to print progress messages.
      Defaults to True.
    - seed (int, optional): Seed for random number generation.
      Defaults to 42.

    Returns:
    None

    '''
    columns = ["game_number", "move_number",
               "board", "move", "legal",
               "stockfish_2", "stockfish_5",
               "stockfish_10",
               "move_quality_2", "move_quality_5",
               "move_quality_10", "prev_ELO", "current_ELO",
               "real", "piece_placement", "active_color",
               "castling_availability", "en_passant",
               "halfmove_clock", "fullmove_number",
               "prev_board", "prev_piece_placement",
               "prev_active_color", "prev_castling_availability",
               "prev_en_passant", "prev_halfmove_clock",
               "prev_fullmove_number"]
    random.seed(seed)
    all_games_df = pd.DataFrame(columns=columns)
    done = False
    i = 0
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    with open(pgn_file_path, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as decompressed_file:
            while not done:
                chunk = decompressed_file.read(1024**3)
                if not chunk:
                    break
                pgn_text = chunk.decode("utf-8")
                pgn_io = io.StringIO(pgn_text)
                while True:
                    pgn_game = chess.pgn.read_game(pgn_io)
                    if i >= max_num_games:
                        done = True
                        break
                    elif pgn_game is None:
                        break

                    all_games_df = save_game_data(all_games_df=all_games_df,
                                                  game_number=i,
                                                  game=pgn_game,
                                                  columns=columns,
                                                  engine=engine)
                    i += 1
    if shuffle:
        all_games_df = all_games_df.sample(
            frac=1, random_state=seed).reset_index(
            drop=True)
    all_games_df.to_csv(save_file_path, index=False, compression="gzip")
    if verbose:
        print(f"Num processed games in a file = {i}")


def get_human_move(data: Dict[str, list],
                   board: chess.Board,
                   move: chess.Move,
                   game_number: int,
                   j: int,
                   engine: chess.engine.SimpleEngine,
                   color: bool,
                   time_limit: float,
                   white_elo: int,
                   black_elo: int) -> Tuple[Dict[str, list], chess.Board]:
    '''
    Get human player's move and update the data.

    Parameters:
    - data (dict): Dictionary containing game data.
    - board (chess.Board): The current state of the chess board.
    - move (chess.Move): The move made by the human player.
    - game_number (int): The number of the game.
    - j (int): The move number.
    - engine (chess.engine.SimpleEngine): The chess engine used for analysis.
    - color (bool): The color of the player making the move.
    - time_limit (float): Time limit for engine analysis in seconds.
    - white_elo (int): White player's Elo rating.
    - black_elo (int): Black player's Elo rating.

    Returns:
    tuple: Updated data dictionary and board state.
    '''
    prev_board = board.fen()
    board.push(move)
    str_representation = board.fen()
    data["game_number"].append(game_number)
    data["move_number"].append(j)
    data["board"].append(str_representation)
    data["prev_board"].append(prev_board)
    data["move"].append(move.uci())
    data["legal"].append(True)
    score2, score5, score10 = get_stockfish_scores(
        board, engine, color, time_limit)
    data["stockfish_2"].append(score2)
    data["stockfish_5"].append(score5)
    data["stockfish_10"].append(score10)
    if j > 2:
        data["move_quality_2"].append(
            data["stockfish_2"][-1] + data["stockfish_2"][-3])
        data["move_quality_5"].append(
            data["stockfish_5"][-1] + data["stockfish_5"][-3])
        data["move_quality_10"].append(
            data["stockfish_10"][-1] + data["stockfish_10"][-3])
    else:
        data["move_quality_2"].append(None)
        data["move_quality_5"].append(None)
        data["move_quality_10"].append(None)
    data["real"].append(True)
    temp_representation = str_representation.split()
    temp_prev_representation = prev_board.split()
    data["piece_placement"].append(temp_representation[0])
    data["active_color"].append(temp_representation[1])
    if temp_representation[1] == "b":
        data["prev_ELO"].append(white_elo)
        data["current_ELO"].append(black_elo)
    else:
        data["prev_ELO"].append(black_elo)
        data["current_ELO"].append(white_elo)
    data["castling_availability"].append(temp_representation[2])
    data["en_passant"].append(temp_representation[3])
    data["halfmove_clock"].append(temp_representation[4])
    data["fullmove_number"].append(temp_representation[5])

    data["prev_piece_placement"].append(temp_prev_representation[0])
    data["prev_active_color"].append(temp_prev_representation[1])
    data["prev_castling_availability"].append(temp_prev_representation[2])
    data["prev_en_passant"].append(temp_prev_representation[3])
    data["prev_halfmove_clock"].append(temp_prev_representation[4])
    data["prev_fullmove_number"].append(temp_prev_representation[5])

    return data, board


def get_random_move(data: Dict[str, list],
                    board: chess.Board,
                    game_number: int,
                    j: int,
                    engine: chess.engine.SimpleEngine,
                    color: bool,
                    time_limit: float,
                    white_elo: int,
                    black_elo: int) -> Tuple[Dict[str, list], chess.Board]:
    '''
    Get a random move and update the data.

    Parameters:
    - data (dict): Dictionary containing game data.
    - board (chess.Board): The current state of the chess board.
    - game_number (int): The number of the game.
    - j (int): The move number.
    - engine (chess.engine.SimpleEngine): The chess engine used for analysis.
    - color (bool): The color of the player making the move.
    - time_limit (float): Time limit for engine analysis in seconds.
    - white_elo (int): White player's Elo rating.
    - black_elo (int): Black player's Elo rating.

    Returns:
    tuple: Updated data dictionary and board state.

    '''
    prev_board = board.fen()
    possible_moves = get_pseudolegal_moves(board)
    legal_moves = list(board.legal_moves)
    random_move = random.choice(possible_moves)
    board.push(random_move)
    str_representation = board.fen()
    data["game_number"].append(game_number)
    data["move_number"].append(j)
    data["board"].append(str_representation)
    data["prev_board"].append(prev_board)
    data["move"].append(random_move.uci())
    if random_move in legal_moves:
        legal = True
    else:
        legal = False
    data["legal"].append(legal)
    if legal:
        score2, score5, score10 = get_stockfish_scores(
            board, engine, color, time_limit)
        data["stockfish_2"].append(score2)
        data["stockfish_5"].append(score5)
        data["stockfish_10"].append(score10)
        if j > 2:
            data["move_quality_2"].append(
                data["stockfish_2"][-1] + data["stockfish_2"][-2])
            data["move_quality_5"].append(
                data["stockfish_5"][-1] + data["stockfish_5"][-2])
            data["move_quality_10"].append(
                data["stockfish_10"][-1] + data["stockfish_10"][-2])
        else:
            data["move_quality_2"].append(None)
            data["move_quality_5"].append(None)
            data["move_quality_10"].append(None)
    else:
        data["stockfish_2"].append(None)
        data["stockfish_5"].append(None)
        data["stockfish_10"].append(None)
        data["move_quality_2"].append(None)
        data["move_quality_5"].append(None)
        data["move_quality_10"].append(None)
    data["real"].append(False)
    temp_representation = str_representation.split()
    temp_prev_representation = prev_board.split()
    data["piece_placement"].append(temp_representation[0])
    data["active_color"].append(temp_representation[1])
    data["castling_availability"].append(temp_representation[2])
    data["en_passant"].append(temp_representation[3])
    data["halfmove_clock"].append(temp_representation[4])
    data["fullmove_number"].append(temp_representation[5])
    data["prev_ELO"].append(None)
    if temp_representation[1] == "b":
        data["current_ELO"].append(black_elo)
    else:
        data["current_ELO"].append(white_elo)

    data["prev_piece_placement"].append(temp_prev_representation[0])
    data["prev_active_color"].append(temp_prev_representation[1])
    data["prev_castling_availability"].append(temp_prev_representation[2])
    data["prev_en_passant"].append(temp_prev_representation[3])
    data["prev_halfmove_clock"].append(temp_prev_representation[4])
    data["prev_fullmove_number"].append(temp_prev_representation[5])

    board.pop()
    return data, board


def get_stockfish_scores(board: chess.Board,
                         engine: chess.engine.SimpleEngine,
                         color: bool,
                         time_limit: float) -> Tuple[int, int, int]:
    '''
    Get evaluation scores from Stockfish for different depths.

    Parameters:
    - board (chess.Board): The current state of the chess board.
    - engine (chess.engine.SimpleEngine): The chess engine used for analysis.
    - color (bool): The color of the player making the move.
    - time_limit (float): Time limit for engine analysis in seconds.

    Returns:
    tuple: Evaluation scores for depths 2, 5, and 10.

    '''
    info_2 = engine.analyse(
        board, chess.engine.Limit(
            depth=2, time=time_limit))
    score_2 = info_2['score'].pov(color=color).score(mate_score=900)
    info_5 = engine.analyse(
        board, chess.engine.Limit(
            depth=5, time=time_limit))
    score_5 = info_5['score'].pov(color=color).score(mate_score=900)
    info_10 = engine.analyse(
        board, chess.engine.Limit(
            depth=10, time=time_limit))
    score_10 = info_10['score'].pov(color=color).score(mate_score=900)
    return score_2, score_5, score_10


def get_pseudolegal_moves(board: chess.Board) -> List[chess.Move]:
    '''
    Ger a list of all possible pseudolegal
    moves for the current board position.

    Parameters:
    - board (chess.Board): The current state of the chess board.

    Returns:
    list: List of all possible pseudolegal moves.

    '''
    pseudolegal_moves = []
    for from_square in chess.SQUARES:
        if board.piece_at(from_square) is not None:
            for to_square in chess.SQUARES:
                pseudolegal_moves.append(chess.Move(from_square, to_square))
    return pseudolegal_moves


def calculate_performance(df: pd.DataFrame) -> None:
    '''
    Calculate and print performance metrics for human moves and random moves.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df['real']]
    random_df = df.loc[not df['real']]
    human_average2 = human_df["stockfish_2"].mean()
    random_average2 = random_df["stockfish_2"].mean()
    human_average5 = human_df["stockfish_5"].mean()
    random_average5 = random_df["stockfish_5"].mean()
    human_average10 = human_df["stockfish_10"].mean()
    random_average10 = random_df["stockfish_10"].mean()

    move_human_average2 = human_df["move_quality_2"].mean()
    move_random_average2 = random_df["move_quality_2"].mean()
    move_human_average5 = human_df["move_quality_5"].mean()
    move_random_average5 = random_df["move_quality_5"].mean()
    move_human_average10 = human_df["move_quality_10"].mean()
    move_random_average10 = random_df["move_quality_10"].mean()

    human_std2 = human_df["stockfish_2"].std()
    random_std2 = random_df["stockfish_2"].std()
    human_std5 = human_df["stockfish_5"].std()
    random_std5 = random_df["stockfish_5"].std()
    human_std10 = human_df["stockfish_10"].std()
    random_std10 = random_df["stockfish_10"].std()

    move_human_std2 = human_df["move_quality_2"].std()
    move_random_std2 = random_df["move_quality_2"].std()
    move_human_std5 = human_df["move_quality_5"].std()
    move_random_std5 = random_df["move_quality_5"].std()
    move_human_std10 = human_df["move_quality_10"].std()
    move_random_std10 = random_df["move_quality_10"].std()

    print(
        f"stockfish_2: Human Mean = {human_average2}, \
        Random Mean = {random_average2}, \
        Human Std = {human_std2}, \
        Random Std = {random_std2}")
    print(
        f"stockfish_5: Human Mean = {human_average5}, \
        Random Mean = {random_average5}, \
        Human Std = {human_std5}, \
        Random Std = {random_std5}")
    print(
        f"stockfish_10: \
        Human Mean = {human_average10}, \
        Random Mean = {random_average10}, \
        Human Std = {human_std10}, \
        Random Std = {random_std10}")
    print(
        f"move_quality_2: \
        Human Mean = {move_human_average2}, \
        Random Mean = {move_random_average2}, \
        Human Std = {move_human_std2}, \
        Random Std = {move_random_std2}")
    print(
        f"move_quality_5: \
        Human Mean = {move_human_average5}, \
        Random Mean = {move_random_average5}, \
        Human Std = {move_human_std5}, \
        Random Std = {move_random_std5}")
    print(
        f"move_quality_10: \
        Human Mean = {move_human_average10}, \
        Random Mean = {move_random_average10}, \
        Human Std = {move_human_std10}, \
        Random Std = {move_random_std10}")


def get_position_quality_histogram(df: pd.DataFrame) -> None:
    '''
    Generate and display a bar chart showing the average
    Stockfish scores for different positions for human and random players.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df['real']]
    random_df = df.loc[not df['real']]
    human_average2 = human_df["stockfish_2"].mean()
    random_average2 = random_df["stockfish_2"].mean()
    human_average5 = human_df["stockfish_5"].mean()
    random_average5 = random_df["stockfish_5"].mean()
    human_average10 = human_df["stockfish_10"].mean()
    random_average10 = random_df["stockfish_10"].mean()
    names = [
        "Human Move - Depth 2",
        "Random Move - Depth 2",
        "Human Move - Depth 5",
        "Random Move - Depth 5",
        "Human Move - Depth 10",
        "Random Move - Depth 10"]
    average_scores = [
        human_average2,
        random_average2,
        human_average5,
        random_average5,
        human_average10,
        random_average10]
    plt.figure(figsize=(10, 8))
    plt.bar(names, average_scores, color="grey")
    plt.xlabel('StockFish Depth and Human/Random Positions', fontsize=14)
    plt.ylabel('Stockfish Score', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Position Quality Barchart', fontsize=16)
    plt.legend()
    plt.show()


def get_move_quality_histogram(df: pd.DataFrame) -> None:
    '''
    Generate and display a bar chart showing
    the average Stockfish scores for different
    moves for human and random players.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df['real']]
    random_df = df.loc[not df['real']]
    human_average2 = human_df["move_quality_2"].mean()
    random_average2 = random_df["move_quality_2"].mean()
    human_average5 = human_df["move_quality_5"].mean()
    random_average5 = random_df["move_quality_5"].mean()
    human_average10 = human_df["move_quality_10"].mean()
    random_average10 = random_df["move_quality_10"].mean()
    names = [
        "Human Move - Depth 2",
        "Random Move - Depth 2",
        "Human Move - Depth 5",
        "Random Move - Depth 5",
        "Human Move - Depth 10",
        "Random Move - Depth 10"]
    average_scores = [
        human_average2,
        random_average2,
        human_average5,
        random_average5,
        human_average10,
        random_average10]
    plt.figure(figsize=(10, 8))
    plt.bar(names, average_scores, color="grey")
    plt.xlabel('StockFish Depth and Human/Random Moves', fontsize=14)
    plt.ylabel('Stockfish Score', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Move Quality Barchart', fontsize=16)
    plt.legend()
    plt.show()


def violin_plot_moves(copied_df: pd.DataFrame) -> None:
    '''
    Generate and display violin plots for move quality.

    Parameters:
    - copied_df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    sns.set_theme(style="darkgrid")
    f, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust plot size
    range_max = 1200
    range_min = -1200

    df = copied_df.copy()
    df['real'] = df['real'].map({True: 'Human Move', False: 'Random Move'})

    sns.violinplot(
        x="real",
        y="move_quality_2",
        hue="real",
        data=df,
        palette="rocket_r",
        ax=axes[0])
    axes[0].set_xlabel('', fontsize=1)
    axes[0].set_ylabel('Move Quality', fontsize=16)
    axes[0].set_title('Move Quality Violin Plot (Depth 2)', fontsize=16)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].set_ylim(range_min, range_max)

    sns.violinplot(
        x="real",
        y="move_quality_5",
        hue="real",
        data=df,
        palette="rocket_r",
        ax=axes[1])
    axes[1].set_xlabel('Move Type', fontsize=16)
    axes[1].set_ylabel('', fontsize=1)
    axes[1].set_title('Move Quality Violin Plot (Depth 5)', fontsize=16)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=0)
    axes[1].legend(fontsize=12)
    axes[1].set_ylim(range_min, range_max)

    sns.violinplot(
        x="real",
        y="move_quality_10",
        hue="real",
        data=df,
        palette="rocket_r",
        ax=axes[2])
    axes[2].set_xlabel('', fontsize=1)
    axes[2].set_ylabel('', fontsize=1)
    axes[2].set_title('Move Quality Violin Plot (Depth 10)', fontsize=16)
    axes[2].tick_params(axis='x', labelsize=12)
    axes[2].tick_params(axis='y', labelsize=0)
    axes[2].legend(fontsize=12)
    axes[2].set_ylim(range_min, range_max)

    plt.tight_layout()
    plt.show()


def violin_plot_positions(copied_df: pd.DataFrame) -> None:
    '''
    Generate and display violin plots for chess position quality.

    Parameters:
    - copied_df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    sns.set_theme(style="darkgrid")
    f, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust plot size
    range_max = 1800
    range_min = -1800

    df = copied_df.copy()
    df['real'] = df['real'].map({True: 'Human Move', False: 'Random Move'})

    sns.violinplot(
        x="real",
        y="stockfish_2",
        hue="real",
        data=df,
        palette="rocket_r",
        ax=axes[0])
    axes[0].set_xlabel('', fontsize=1)
    axes[0].set_ylabel('Position Quality', fontsize=16)
    axes[0].set_title('Position Quality Violin Plot (Depth 2)', fontsize=16)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].set_ylim(range_min, range_max)

    sns.violinplot(
        x="real",
        y="stockfish_5",
        hue="real",
        data=df,
        palette="rocket_r",
        ax=axes[1])
    axes[1].set_xlabel('Position Type', fontsize=16)
    axes[1].set_ylabel('', fontsize=1)
    axes[1].set_title('Position Quality Violin Plot (Depth 5)', fontsize=16)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=0)
    axes[1].legend(fontsize=12)
    axes[1].set_ylim(range_min, range_max)

    sns.violinplot(
        x="real",
        y="stockfish_10",
        hue="real",
        data=df,
        palette="rocket_r",
        ax=axes[2])
    axes[2].set_xlabel('', fontsize=1)
    axes[2].set_ylabel('', fontsize=1)
    axes[2].set_title('Position Quality Violin Plot (Depth 10)', fontsize=16)
    axes[2].tick_params(axis='x', labelsize=12)
    axes[2].tick_params(axis='y', labelsize=0)
    axes[2].legend(fontsize=12)
    axes[2].set_ylim(range_min, range_max)

    plt.tight_layout()
    plt.show()


def plot_real_fake(df: pd.DataFrame) -> None:
    '''
    Generate and display a
    bar chart comparing the counts
    of real and random moves.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    real_moves = df.loc[df["real"]]
    fake_moves = df.loc[not df["real"]]
    values = [len(real_moves), len(fake_moves)]
    names = ["Real Moves", "Random Moves"]
    plt.figure(figsize=(4, 6))
    plt.bar(names, values, color="grey")
    plt.xlabel('Moves', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Number of Legal vs Illegal Moves', fontsize=16)
    plt.legend()
    plt.show()


def plot_legal_illegal(df: pd.DataFrame) -> None:
    '''
    Create and showcase a bar chart
    comparing the counts of legal and
    illegal moves.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    legal_moves = df.loc[df["legal"]]
    illegal_moves = df.loc[not df["legal"]]
    values = [len(illegal_moves), len(legal_moves)]
    names = ["Illegal Moves", "Legal Moves"]
    plt.figure(figsize=(4, 6))
    plt.bar(names, values, color="grey")
    plt.xlabel('Moves', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Number of Legal vs Illegal Moves', fontsize=16)
    plt.legend()
    plt.show()


def plot_most_common_moves(temp_df: pd.DataFrame) -> None:
    '''
    Create and showcase a bar chart
    showing the most common human
    moves.

    Parameters:
    - temp_df (DataFrame): DataFrame
    containing game data.

    Returns:
    None

    '''
    df = temp_df.loc[temp_df["real"]]
    top_common_moves = df["move"].value_counts().head(10)
    values = top_common_moves.values
    names = top_common_moves.index
    plt.figure(figsize=(10, 8))
    plt.bar(names, values, color="grey")
    plt.xlabel('Most Common Moves', fontsize=14)
    plt.ylabel('Number of Occurences', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Most Common Human Moves', fontsize=16)
    plt.legend()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    '''
    Create and showcase a heatmap showing correlations
    between numeric features in the data.

    Parameters:
    - df (DataFrame): DataFrame containing
    game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['current_ELO'] = pd.to_numeric(
        human_df['current_ELO'], errors='coerce')
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    human_df = human_df.select_dtypes(include='number')
    corr_matrix = human_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5)
    plt.title('Correlations of Data Features for Chess')
    plt.show()


def plot_elo_move_quality(df: pd.DataFrame) -> None:
    '''
    Create and showcase a set of scatter plots
    showing the relationship between
    player's ELO score
    and move quality
    for different depths.

    Parameters:
    - df (DataFrame): DataFrame
    containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['current_ELO'] = pd.to_numeric(
        human_df['current_ELO'], errors='coerce')
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    f, axes = plt.subplots(1, 3, figsize=(18, 6))
    human_df = human_df.sample(n=1000, random_state=42)
    range_max = 1200
    range_min = -1200

    axes[0].scatter(human_df["prev_ELO"], human_df["move_quality_2"])
    axes[0].set_xlabel('', fontsize=1)
    axes[0].set_ylabel('Move Quality', fontsize=16)
    axes[0].set_title(
        'Move Quality vs Player\'s ELO Score (Depth 2)',
        fontsize=16)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].set_ylim(range_min, range_max)

    axes[1].scatter(human_df["prev_ELO"], human_df["move_quality_5"])
    axes[1].set_xlabel('Player\'s ELO score', fontsize=16)
    axes[1].set_ylabel('', fontsize=1)
    axes[1].set_title(
        'Move Quality vs Player\'s ELO Score (Depth 5)',
        fontsize=16)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=0)
    axes[1].legend(fontsize=12)
    axes[1].set_ylim(range_min, range_max)

    axes[2].scatter(human_df["prev_ELO"], human_df["move_quality_10"])
    axes[2].set_xlabel('', fontsize=1)
    axes[2].set_ylabel('', fontsize=1)
    axes[2].set_title(
        'Move Quality vs Player\'s ELO Score (Depth 10)',
        fontsize=16)
    axes[2].tick_params(axis='x', labelsize=12)
    axes[2].tick_params(axis='y', labelsize=0)
    axes[2].legend(fontsize=12)
    axes[2].set_ylim(range_min, range_max)

    plt.tight_layout()
    plt.show()


def plot_elo_position_quality(df: pd.DataFrame) -> None:
    '''
    Create and showcase a set of scatter plots showing
    the relationship between player's ELO score and
    position quality for different depths.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['current_ELO'] = pd.to_numeric(
        human_df['current_ELO'], errors='coerce')
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    f, axes = plt.subplots(1, 3, figsize=(18, 6))
    human_df = human_df.sample(n=1000, random_state=42)
    range_max = 2400
    range_min = -2400

    axes[0].scatter(human_df["prev_ELO"], human_df["stockfish_2"])
    axes[0].set_xlabel('', fontsize=1)
    axes[0].set_ylabel('Position Quality', fontsize=16)
    axes[0].set_title(
        'Position Quality vs Player\'s ELO Score (Depth 2)',
        fontsize=16)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].set_ylim(range_min, range_max)

    axes[1].scatter(human_df["prev_ELO"], human_df["stockfish_5"])
    axes[1].set_xlabel('Player\'s ELO score', fontsize=16)
    axes[1].set_ylabel('', fontsize=1)
    axes[1].set_title(
        'Position Quality vs Player\'s ELO Score (Depth 5)',
        fontsize=16)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=0)
    axes[1].legend(fontsize=12)
    axes[1].set_ylim(range_min, range_max)

    axes[2].scatter(human_df["prev_ELO"], human_df["stockfish_10"])
    axes[2].set_xlabel('', fontsize=1)
    axes[2].set_ylabel('', fontsize=1)
    axes[2].set_title(
        'Position Quality vs Player\'s ELO Score (Depth 10)',
        fontsize=16)
    axes[2].tick_params(axis='x', labelsize=12)
    axes[2].tick_params(axis='y', labelsize=0)
    axes[2].legend(fontsize=12)
    axes[2].set_ylim(range_min, range_max)

    plt.tight_layout()
    plt.show()


def plot_ELO_move_quality_3D(df: pd.DataFrame) -> None:
    '''
    Create and showcase a 3D scatter plot
    showing the relationship between player's ELO
    score and move quality for different depths.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    human_df = human_df.sample(n=10000, random_state=42)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(
        human_df["move_quality_2"],
        human_df["move_quality_5"],
        human_df["move_quality_10"],
        c=human_df["prev_ELO"],
        cmap='coolwarm',
        marker='o')
    ax.set_xlabel('Move Quality (Depth 2)', fontsize=14)
    ax.set_ylabel('Move Quality (Depth 5)', fontsize=14)
    ax.set_zlabel('Move Quality (Depth 10)', fontsize=14)
    plt.title('Move Quality vs Player\'s ELO', fontsize=16)
    plt.colorbar(sc, label='Player\'s ELO')
    plt.show()


def plot_ELO_position_quality_3D(df: pd.DataFrame) -> None:
    '''
    Create and showcase a 3D scatter plot showing the
    relationship between player's ELO score and position
    quality for different depths.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['current_ELO'] = pd.to_numeric(
        human_df['current_ELO'], errors='coerce')
    human_df = human_df.sample(n=10000, random_state=42)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(
        human_df["stockfish_2"],
        human_df["stockfish_5"],
        human_df["stockfish_10"],
        c=human_df["current_ELO"],
        cmap='coolwarm',
        marker='o')
    ax.set_xlabel('Position Quality (Depth 2)', fontsize=14)
    ax.set_ylabel('Position Quality (Depth 5)', fontsize=14)
    ax.set_zlabel('Position Quality (Depth 10)', fontsize=14)
    plt.title('Position Quality vs Player\'s ELO', fontsize=16)
    plt.colorbar(sc, label='Player\'s ELO')
    plt.show()


def plot_most_common_moves_per_ELO(df: pd.DataFrame) -> None:
    '''
    Create and showcase a set of bar charts showing
    the most common moves for players at different
    ELO levels.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    bins = [0, 1000, 1500, 2000, 3000]
    labels = ['Novice', 'Intermidiate', 'Expert', 'Master']
    human_df['player_level'] = pd.cut(
        human_df['prev_ELO'],
        bins=bins,
        labels=labels,
        right=False)
    novice_players = human_df.loc[human_df["player_level"] == 'Novice']
    intermidiate_players = human_df.loc[human_df["player_level"]
                                        == 'Intermidiate']
    expert_players = human_df.loc[human_df["player_level"] == 'Expert']
    master_players = human_df.loc[human_df["player_level"] == 'Master']
    top_novice_moves = novice_players["move"].value_counts().head(10)
    top_intermidiate_moves = intermidiate_players["move"].value_counts().head(
        10)
    top_expert_moves = expert_players["move"].value_counts().head(10)
    top_master_moves = master_players["move"].value_counts().head(10)
    f, axes = plt.subplots(2, 2, figsize=(12, 12))

    novice_values = top_novice_moves.values
    novice_names = top_novice_moves.index
    intermidiate_values = top_intermidiate_moves.values
    intermidiate_names = top_intermidiate_moves.index
    expert_values = top_expert_moves.values
    expert_names = top_expert_moves.index
    master_values = top_master_moves.values
    master_names = top_master_moves.index

    axes[0, 0].bar(novice_names, novice_values, color="grey")
    axes[0, 0].set_xlabel('Moves', fontsize=14)
    axes[0, 0].set_ylabel('Number of Occurences', fontsize=14)
    axes[0, 0].set_title('Most Common Moves for Novices', fontsize=16)
    axes[0, 0].tick_params(axis='x', labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)

    axes[1, 0].bar(intermidiate_names, intermidiate_values, color="grey")
    axes[1, 0].set_xlabel('Moves', fontsize=14)
    axes[1, 0].set_ylabel('Number of Occurences', fontsize=14)
    axes[1, 0].set_title(
        'Most Common Moves for Intermidiate Players', fontsize=16)
    axes[1, 0].tick_params(axis='x', labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=12)

    axes[0, 1].bar(expert_names, expert_values, color="grey")
    axes[0, 1].set_xlabel('Moves', fontsize=14)
    axes[0, 1].set_ylabel('Number of Occurences', fontsize=14)
    axes[0, 1].set_title('Most Common Moves for Expert Players', fontsize=16)
    axes[0, 1].tick_params(axis='x', labelsize=12)
    axes[0, 1].tick_params(axis='y', labelsize=12)

    axes[1, 1].bar(master_names, master_values, color="grey")
    axes[1, 1].set_xlabel('Moves', fontsize=14)
    axes[1, 1].set_ylabel('Number of Occurences', fontsize=14)
    axes[1, 1].set_title('Most Common Moves for Master Players', fontsize=16)
    axes[1, 1].tick_params(axis='x', labelsize=12)
    axes[1, 1].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_most_common_moves_per_ELO_relative(df: pd.DataFrame) -> None:
    '''
    Create and showcase a set of bar charts
    showing the relative frequency of the most
    common moves for players at different ELO levels.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    bins = [0, 1000, 1500, 2000, 3000]
    labels = ['Novice', 'Intermidiate', 'Expert', 'Master']
    human_df['player_level'] = pd.cut(
        human_df['prev_ELO'],
        bins=bins,
        labels=labels,
        right=False)
    novice_players = human_df.loc[human_df["player_level"] == 'Novice']
    intermidiate_players = human_df.loc[human_df["player_level"]
                                        == 'Intermidiate']
    expert_players = human_df.loc[human_df["player_level"] == 'Expert']
    master_players = human_df.loc[human_df["player_level"] == 'Master']
    top_novice_moves = novice_players["move"].value_counts().head(10)
    top_intermidiate_moves = intermidiate_players["move"].value_counts().head(
        10)
    top_expert_moves = expert_players["move"].value_counts().head(10)
    top_master_moves = master_players["move"].value_counts().head(10)
    f, axes = plt.subplots(2, 2, figsize=(12, 12))

    novice_values = top_novice_moves.values
    novice_names = top_novice_moves.index
    intermidiate_values = top_intermidiate_moves.values
    intermidiate_names = top_intermidiate_moves.index
    expert_values = top_expert_moves.values
    expert_names = top_expert_moves.index
    master_values = top_master_moves.values
    master_names = top_master_moves.index

    novice_values = np.array(novice_values)
    intermidiate_values = np.array(intermidiate_values)
    expert_values = np.array(expert_values)
    master_values = np.array(master_values)
    novice_values = novice_values / len(novice_players)
    intermidiate_values = intermidiate_values / len(intermidiate_players)
    expert_values = expert_values / len(expert_players)
    master_values = master_values / len(master_players)

    axes[0, 0].bar(novice_names, novice_values, color="grey")
    axes[0, 0].set_xlabel('Moves', fontsize=14)
    axes[0, 0].set_ylabel('Frequency', fontsize=14)
    axes[0, 0].set_title('Most Common Moves for Novices', fontsize=16)
    axes[0, 0].tick_params(axis='x', labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)

    axes[1, 0].bar(intermidiate_names, intermidiate_values, color="grey")
    axes[1, 0].set_xlabel('Moves', fontsize=14)
    axes[1, 0].set_ylabel('Frequency', fontsize=14)
    axes[1, 0].set_title(
        'Most Common Moves for Intermidiate Players', fontsize=16)
    axes[1, 0].tick_params(axis='x', labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=12)

    axes[0, 1].bar(expert_names, expert_values, color="grey")
    axes[0, 1].set_xlabel('Moves', fontsize=14)
    axes[0, 1].set_ylabel('Frequency', fontsize=14)
    axes[0, 1].set_title('Most Common Moves for Expert Players', fontsize=16)
    axes[0, 1].tick_params(axis='x', labelsize=12)
    axes[0, 1].tick_params(axis='y', labelsize=12)

    axes[1, 1].bar(master_names, master_values, color="grey")
    axes[1, 1].set_xlabel('Moves', fontsize=14)
    axes[1, 1].set_ylabel('Frequency', fontsize=14)
    axes[1, 1].set_title('Most Common Moves for Master Players', fontsize=16)
    axes[1, 1].tick_params(axis='x', labelsize=12)
    axes[1, 1].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_most_common_moves_per_ELO_colors(df: pd.DataFrame) -> None:
    '''
    Create and showcase a set of different color
    bar charts showing the most common moves for
    players at different ELO levels, with each ELO
    level represented by a different color.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    bins = [0, 1000, 1500, 2000, 3000]
    labels = ['Novice', 'Intermidiate', 'Expert', 'Master']
    human_df['player_level'] = pd.cut(
        human_df['prev_ELO'],
        bins=bins,
        labels=labels,
        right=False)
    novice_players = human_df.loc[human_df["player_level"] == 'Novice']
    intermidiate_players = human_df.loc[human_df["player_level"]
                                        == 'Intermidiate']
    expert_players = human_df.loc[human_df["player_level"] == 'Expert']
    master_players = human_df.loc[human_df["player_level"] == 'Master']
    top_novice_moves = novice_players["move"].value_counts().head(10)
    top_intermidiate_moves = intermidiate_players["move"].value_counts().head(
        10)
    top_expert_moves = expert_players["move"].value_counts().head(10)
    top_master_moves = master_players["move"].value_counts().head(10)
    f, axes = plt.subplots(2, 2, figsize=(12, 12))

    novice_values = top_novice_moves.values
    novice_names = top_novice_moves.index
    intermidiate_values = top_intermidiate_moves.values
    intermidiate_names = top_intermidiate_moves.index
    expert_values = top_expert_moves.values
    expert_names = top_expert_moves.index
    master_values = top_master_moves.values
    master_names = top_master_moves.index

    axes[0, 0].bar(novice_names, novice_values, color="green")
    axes[0, 0].set_xlabel('Moves', fontsize=14)
    axes[0, 0].set_ylabel('Number of Occurences', fontsize=14)
    axes[0, 0].set_title('Most Common Moves for Novices', fontsize=16)
    axes[0, 0].tick_params(axis='x', labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)

    axes[1, 0].bar(intermidiate_names, intermidiate_values, color="blue")
    axes[1, 0].set_xlabel('Moves', fontsize=14)
    axes[1, 0].set_ylabel('Number of Occurences', fontsize=14)
    axes[1, 0].set_title(
        'Most Common Moves for Intermidiate Players', fontsize=16)
    axes[1, 0].tick_params(axis='x', labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=12)

    axes[0, 1].bar(expert_names, expert_values, color="orange")
    axes[0, 1].set_xlabel('Moves', fontsize=14)
    axes[0, 1].set_ylabel('Number of Occurences', fontsize=14)
    axes[0, 1].set_title('Most Common Moves for Expert Players', fontsize=16)
    axes[0, 1].tick_params(axis='x', labelsize=12)
    axes[0, 1].tick_params(axis='y', labelsize=12)

    axes[1, 1].bar(master_names, master_values, color="red")
    axes[1, 1].set_xlabel('Moves', fontsize=14)
    axes[1, 1].set_ylabel('Number of Occurences', fontsize=14)
    axes[1, 1].set_title('Most Common Moves for Master Players', fontsize=16)
    axes[1, 1].tick_params(axis='x', labelsize=12)
    axes[1, 1].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_most_common_moves_per_ELO_relative_colors(df: pd.DataFrame) -> None:
    '''
    Create and showcase a set of bar charts
    showing the relative frequency of the most
    common moves for players at different ELO levels,
    with each ELO level represented by a different color.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    bins = [0, 1000, 1500, 2000, 3000]
    labels = ['Novice', 'Intermidiate', 'Expert', 'Master']
    human_df['player_level'] = pd.cut(
        human_df['prev_ELO'],
        bins=bins,
        labels=labels,
        right=False)
    novice_players = human_df.loc[human_df["player_level"] == 'Novice']
    intermidiate_players = human_df.loc[human_df["player_level"]
                                        == 'Intermidiate']
    expert_players = human_df.loc[human_df["player_level"] == 'Expert']
    master_players = human_df.loc[human_df["player_level"] == 'Master']
    top_novice_moves = novice_players["move"].value_counts().head(10)
    top_intermidiate_moves = intermidiate_players["move"].value_counts().head(
        10)
    top_expert_moves = expert_players["move"].value_counts().head(10)
    top_master_moves = master_players["move"].value_counts().head(10)
    f, axes = plt.subplots(2, 2, figsize=(12, 12))

    novice_values = top_novice_moves.values
    novice_names = top_novice_moves.index
    intermidiate_values = top_intermidiate_moves.values
    intermidiate_names = top_intermidiate_moves.index
    expert_values = top_expert_moves.values
    expert_names = top_expert_moves.index
    master_values = top_master_moves.values
    master_names = top_master_moves.index

    novice_values = np.array(novice_values)
    intermidiate_values = np.array(intermidiate_values)
    expert_values = np.array(expert_values)
    master_values = np.array(master_values)
    novice_values = novice_values / len(novice_players)
    intermidiate_values = intermidiate_values / len(intermidiate_players)
    expert_values = expert_values / len(expert_players)
    master_values = master_values / len(master_players)

    axes[0, 0].bar(novice_names, novice_values, color="green")
    axes[0, 0].set_xlabel('Moves', fontsize=14)
    axes[0, 0].set_ylabel('Frequency', fontsize=14)
    axes[0, 0].set_title('Most Common Moves for Novices', fontsize=16)
    axes[0, 0].tick_params(axis='x', labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)

    axes[1, 0].bar(intermidiate_names, intermidiate_values, color="blue")
    axes[1, 0].set_xlabel('Moves', fontsize=14)
    axes[1, 0].set_ylabel('Frequency', fontsize=14)
    axes[1, 0].set_title(
        'Most Common Moves for Intermidiate Players', fontsize=16)
    axes[1, 0].tick_params(axis='x', labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=12)

    axes[0, 1].bar(expert_names, expert_values, color="orange")
    axes[0, 1].set_xlabel('Moves', fontsize=14)
    axes[0, 1].set_ylabel('Frequency', fontsize=14)
    axes[0, 1].set_title('Most Common Moves for Expert Players', fontsize=16)
    axes[0, 1].tick_params(axis='x', labelsize=12)
    axes[0, 1].tick_params(axis='y', labelsize=12)

    axes[1, 1].bar(master_names, master_values, color="red")
    axes[1, 1].set_xlabel('Moves', fontsize=14)
    axes[1, 1].set_ylabel('Frequency', fontsize=14)
    axes[1, 1].set_title('Most Common Moves for Master Players', fontsize=16)
    axes[1, 1].tick_params(axis='x', labelsize=12)
    axes[1, 1].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_most_common_move_evaluations(df: pd.DataFrame) -> None:
    '''
    Create and showcase a set of bar charts showing
    the average evaluation of the most common moves
    for players at different ELO levels.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    bins = [0, 1000, 1500, 2000, 3000]
    labels = ['Novice', 'Intermidiate', 'Expert', 'Master']
    human_df['player_level'] = pd.cut(
        human_df['prev_ELO'],
        bins=bins,
        labels=labels,
        right=False)
    novice_players = human_df.loc[human_df["player_level"] == 'Novice']
    intermidiate_players = human_df.loc[human_df["player_level"]
                                        == 'Intermidiate']
    expert_players = human_df.loc[human_df["player_level"] == 'Expert']
    master_players = human_df.loc[human_df["player_level"] == 'Master']
    top_novice_moves = novice_players["move"].value_counts().head(10)
    top_intermidiate_moves = intermidiate_players["move"].value_counts().head(
        10)
    top_expert_moves = expert_players["move"].value_counts().head(10)
    top_master_moves = master_players["move"].value_counts().head(10)
    f, axes = plt.subplots(2, 2, figsize=(12, 12))

    novice_names = top_novice_moves.index
    intermidiate_names = top_intermidiate_moves.index
    expert_names = top_expert_moves.index
    master_names = top_master_moves.index

    novice_values = []
    for move in novice_names:
        temp_df = novice_players.loc[novice_players["move"] == move]
        novice_values.append(temp_df["move_quality_2"].mean())

    intermidiate_values = []
    for move in intermidiate_names:
        temp_df = intermidiate_players.\
            loc[intermidiate_players["move"] == move]
        intermidiate_values.append(temp_df["move_quality_2"].mean())

    expert_values = []
    for move in expert_names:
        temp_df = expert_players.loc[expert_players["move"] == move]
        expert_values.append(temp_df["move_quality_2"].mean())

    master_values = []
    for move in master_names:
        temp_df = master_players.loc[master_players["move"] == move]
        master_values.append(temp_df["move_quality_2"].mean())

    novice_values.sort()
    intermidiate_values.sort()
    expert_values.sort()
    master_values.sort()

    axes[0, 0].bar(novice_names, novice_values, color="grey")
    axes[0, 0].set_xlabel('Moves', fontsize=14)
    axes[0, 0].set_ylabel('Move Quality', fontsize=14)
    axes[0, 0].set_title(
        'Quality of Most Common Moves for Novices', fontsize=16)
    axes[0, 0].tick_params(axis='x', labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    axes[0, 0].axhline(y=0, color='black', linestyle='--')

    axes[1, 0].bar(intermidiate_names, intermidiate_values, color="grey")
    axes[1, 0].set_xlabel('Moves', fontsize=14)
    axes[1, 0].set_ylabel('Move Quality', fontsize=14)
    axes[1, 0].set_title(
        'Quality of  Most Common Moves for Intermidiate Players', fontsize=16)
    axes[1, 0].tick_params(axis='x', labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=12)

    axes[0, 1].bar(expert_names, expert_values, color="grey")
    axes[0, 1].set_xlabel('Moves', fontsize=14)
    axes[0, 1].set_ylabel('Move Quality', fontsize=14)
    axes[0, 1].set_title(
        'Quality of Most Common Moves for Expert Players', fontsize=16)
    axes[0, 1].tick_params(axis='x', labelsize=12)
    axes[0, 1].tick_params(axis='y', labelsize=12)

    axes[1, 1].bar(master_names, master_values, color="grey")
    axes[1, 1].set_xlabel('Moves', fontsize=14)
    axes[1, 1].set_ylabel('Move Quality', fontsize=14)
    axes[1, 1].set_title(
        'Quality of Most Common Moves for Master Players', fontsize=16)
    axes[1, 1].tick_params(axis='x', labelsize=12)
    axes[1, 1].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_most_common_move_evaluations_color(df: pd.DataFrame) -> None:
    '''
    Create and showcase a set of different color
    bar charts showing the average evaluation of the
    most common moves for players at different ELO levels,
    with each ELO level represented by a different color.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    human_df = df.loc[df["real"]]
    human_df['prev_ELO'] = pd.to_numeric(human_df['prev_ELO'], errors='coerce')
    bins = [0, 1000, 1500, 2000, 3000]
    labels = ['Novice', 'Intermidiate', 'Expert', 'Master']
    human_df['player_level'] = pd.cut(
        human_df['prev_ELO'],
        bins=bins,
        labels=labels,
        right=False)
    novice_players = human_df.loc[human_df["player_level"] == 'Novice']
    intermidiate_players = human_df.loc[human_df["player_level"]
                                        == 'Intermidiate']
    expert_players = human_df.loc[human_df["player_level"] == 'Expert']
    master_players = human_df.loc[human_df["player_level"] == 'Master']
    top_novice_moves = novice_players["move"].value_counts().head(10)
    top_intermidiate_moves = intermidiate_players["move"].value_counts().head(
        10)
    top_expert_moves = expert_players["move"].value_counts().head(10)
    top_master_moves = master_players["move"].value_counts().head(10)
    f, axes = plt.subplots(2, 2, figsize=(12, 12))

    novice_names = top_novice_moves.index
    intermidiate_names = top_intermidiate_moves.index
    expert_names = top_expert_moves.index
    master_names = top_master_moves.index

    novice_values = []
    for move in novice_names:
        temp_df = novice_players.loc[novice_players["move"] == move]
        novice_values.append(temp_df["move_quality_2"].mean())

    intermidiate_values = []
    for move in intermidiate_names:
        temp_df = intermidiate_players.\
            loc[intermidiate_players["move"] == move]
        intermidiate_values.append(temp_df["move_quality_2"].mean())

    expert_values = []
    for move in expert_names:
        temp_df = expert_players.loc[expert_players["move"] == move]
        expert_values.append(temp_df["move_quality_2"].mean())

    master_values = []
    for move in master_names:
        temp_df = master_players.loc[master_players["move"] == move]
        master_values.append(temp_df["move_quality_2"].mean())

    novice_values.sort()
    intermidiate_values.sort()
    expert_values.sort()
    master_values.sort()

    axes[0, 0].bar(novice_names, novice_values, color="green")
    axes[0, 0].set_xlabel('Moves', fontsize=14)
    axes[0, 0].set_ylabel('Move Quality', fontsize=14)
    axes[0, 0].set_title(
        'Quality of Most Common Moves for Novices', fontsize=16)
    axes[0, 0].tick_params(axis='x', labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    axes[0, 0].axhline(y=0, color='black', linestyle='--')

    axes[1, 0].bar(intermidiate_names, intermidiate_values, color="blue")
    axes[1, 0].set_xlabel('Moves', fontsize=14)
    axes[1, 0].set_ylabel('Move Quality', fontsize=14)
    axes[1, 0].set_title(
        'Quality of  Most Common Moves for Intermidiate Players', fontsize=16)
    axes[1, 0].tick_params(axis='x', labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=12)

    axes[0, 1].bar(expert_names, expert_values, color="orange")
    axes[0, 1].set_xlabel('Moves', fontsize=14)
    axes[0, 1].set_ylabel('Move Quality', fontsize=14)
    axes[0, 1].set_title(
        'Quality of Most Common Moves for Expert Players', fontsize=16)
    axes[0, 1].tick_params(axis='x', labelsize=12)
    axes[0, 1].tick_params(axis='y', labelsize=12)

    axes[1, 1].bar(master_names, master_values, color="red")
    axes[1, 1].set_xlabel('Moves', fontsize=14)
    axes[1, 1].set_ylabel('Move Quality', fontsize=14)
    axes[1, 1].set_title(
        'Quality of Most Common Moves for Master Players', fontsize=16)
    axes[1, 1].tick_params(axis='x', labelsize=12)
    axes[1, 1].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_move_quality__random_human_3D(df: pd.DataFrame) -> None:
    '''
    Create and showcase a 3D scatter plot comparing move
    quality (depth 2, 5, and 10) between player moves and
    random moves.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    legal_df = df.loc[df["legal"]]
    real_legal_df = legal_df.loc[legal_df["real"]]
    fake_legal_df = legal_df.loc[not legal_df["real"]]
    real_legal_df = real_legal_df.sample(n=2000, random_state=42)
    fake_legal_df = fake_legal_df.sample(n=2000, random_state=42)
    legal_df = pd.concat([real_legal_df, fake_legal_df], axis=0)
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(projection='3d')
    marker = legal_df["real"].map({True: 'o', False: '^'})
    move_quality_2 = list(legal_df["move_quality_2"].values)
    move_quality_5 = list(legal_df["move_quality_5"].values)
    move_quality_10 = list(legal_df["move_quality_10"].values)

    marker = list(marker.values)
    colors = {'o': "blue", "^": "orange"}
    for i in range(len(legal_df)):
        ax.scatter(move_quality_2[i],
                   move_quality_5[i],
                   move_quality_10[i],
                   marker=marker[i],
                   color=colors[marker[i]])

    ax.set_xlabel('Move Quality (Depth 2)', fontsize=14)
    ax.set_ylabel('Move Quality (Depth 5)', fontsize=14)
    ax.set_zlabel('Move Quality (Depth 10)', fontsize=14)
    plt.title('Move Quality Player vs Random Moves', fontsize=16)
    plt.show()


def plot_position_quality__random_human_3D(df: pd.DataFrame) -> None:
    '''
    Create and showcase a 3D scatter plot comparing
    position quality (depth 2, 5, and 10) between player
    moves and random moves.

    Parameters:
    - df (DataFrame): DataFrame containing game data.

    Returns:
    None

    '''
    legal_df = df.loc[df["legal"]]
    real_legal_df = legal_df.loc[legal_df["real"]]
    fake_legal_df = legal_df.loc[not legal_df["real"]]
    real_legal_df = real_legal_df.sample(n=2000, random_state=42)
    fake_legal_df = fake_legal_df.sample(n=2000, random_state=42)
    legal_df = pd.concat([real_legal_df, fake_legal_df], axis=0)
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(projection='3d')
    marker = legal_df["real"].map({True: 'o', False: '^'})
    move_quality_2 = list(legal_df["stockfish_2"].values)
    move_quality_5 = list(legal_df["stockfish_5"].values)
    move_quality_10 = list(legal_df["stockfish_10"].values)

    marker = list(marker.values)
    colors = {'o': "blue", "^": "orange"}
    for i in range(len(legal_df)):
        ax.scatter(move_quality_2[i],
                   move_quality_5[i],
                   move_quality_10[i],
                   marker=marker[i],
                   color=colors[marker[i]])

    ax.set_xlabel('Position Quality (Depth 2)', fontsize=14)
    ax.set_ylabel('Position Quality (Depth 5)', fontsize=14)
    ax.set_zlabel('Position Quality (Depth 10)', fontsize=14)
    plt.title('Position Quality Player vs Random Moves', fontsize=16)
    plt.show()
