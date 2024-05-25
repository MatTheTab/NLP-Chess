import random

import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, TextGenerationPipeline

import chess
import chess.engine
import chess.svg
from chess import IllegalMoveError as IllMoveError
from chess import InvalidMoveError, AmbiguousMoveError

from tqdm import tqdm
from copy import deepcopy
from math import ceil

from IPython.display import SVG, display

from typing import List, Dict, Tuple, Optional, Any

import warnings
warnings.filterwarnings("ignore")


def show_board(board: chess.Board, size: int = 400) -> None:
    """
    Display the chess board.

    Parameters:
    - board (chess.Board): The current state of the chess board.
    - size (int): Size of the chess board for display. Default is 400.
    """
    display(SVG(chess.svg.board(board=board, size=size)))


def get_n_moves(game_transcript: str, n: int = 10) -> Optional[str]:
    """
    Get the first n moves from the game transcript.

    Parameters:
    - game_transcript (str): The transcript of the chess game.
    - n (int): Number of moves to extract. Default is 10.

    Returns:
    - Optional[str]: The first n moves from the transcript or None if
    index out of range.
    """

    n_games = n + ceil(n / 2)
    try:
        return " ".join(game_transcript.split(" ")[:n_games])
    except IndexError:
        return None


def get_state_after_moves(game_transcript: str, n: int = 10)\
        -> Tuple[Optional[chess.Board], Optional[str], Optional[str]]:
    """
    Get the state of the chess board after n moves.

    Parameters:
    - game_transcript (str): The transcript of the chess game.
    - n (int): Number of moves to play out. Default is 10.

    Returns:
    - Tuple[Optional[chess.Board], Optional[str], Optional[str]]:
    The chess board, game string, and player's next move or None in
    case the game is finished. Returns None, None, None in case of
    invalid game string or if the game is too short.
    """

    board = chess.Board()
    game_str = get_n_moves(game_transcript, n)
    if game_str is None:
        print("No game str")
        return None, None, None

    game = [x for x in game_str.split(" ") if x[0].isalpha()]
    for move in game:
        try:
            board.push_san(move)
        except (IllMoveError, InvalidMoveError, AmbiguousMoveError):
            break

    if len(board.move_stack) != n:  # Not enough moves in the game
        return None, None, None

    player_move = get_n_moves(game_transcript, n + 1)
    if player_move is None:
        return board, game_str, None
    player_move = player_move.split(" ")[-1]

    return board, game_str, player_move


def generate_answer(generator: TextGenerationPipeline,
                    prompt: str,
                    max_len: int = 40,
                    num_answers: int = 1,
                    alternatives: Optional[List[str]] = None) -> List[str]:
    """
    Generate a text answer using the provided generator.

    Parameters:
    - generator (TextGenerationPipeline): The text generation model.
    - prompt (str): The prompt to generate text from.
    - max_len (int): Maximum length of the generated text. Default is 40.
    - num_answers (int): Number of answers to generate. Default is 1.
    - alternatives (Optional[List[str]]): List of alternative words to
    force in the generation.

    Returns:
    - List[str]: The generated text answers.
    """

    if alternatives is None:
        answer = generator(
            prompt,
            max_length=max_len,
            num_return_sequences=num_answers,
            truncation=True,
            pad_token_id=generator.tokenizer.eos_token_id)
    else:
        answer = generator(
            prompt,
            max_length=max_len,
            num_return_sequences=num_answers,
            truncation=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            force_words=alternatives,
            num_beams=4)

    return [x["generated_text"] for x in answer]


def generate_next_moves(moves: str,
                        generator: TextGenerationPipeline,
                        tokenizer: AutoTokenizer,
                        answer_length: int = 20,
                        num_answers: int = 1,
                        alternatives: Optional[List[str]] = None) -> List[str]:
    """
    Generate the next moves in the chess game using a text generation model.

    Parameters:
    - moves (str): The current moves in the chess game.
    - generator (TextGenerationPipeline): The text generation model.
    - tokenizer (AutoTokenizer): The tokenizer for the model.
    - answer_length (int): Length of the answer to generate. Default is 20.
    - num_answers (int): Number of answers to generate. Default is 1.
    - alternatives (Optional[List[str]]): List of alternative words to force
    in the generation.

    Returns:
    - List[str]: The generated next moves in the game.
    """
    query = f"{moves}"
    tokens = tokenizer.tokenize(query)

    answers = generate_answer(
        generator,
        query,
        max_len=len(tokens) +
        answer_length,
        num_answers=num_answers,
        alternatives=alternatives)
    return [answer[len(query) + 1:].strip() for answer in answers]


def get_san_legal_moves(board: chess.Board) -> List[str]:
    """
    Get all legal moves from the board in SAN format.

    Parameters:
    - board (chess.Board): The current state of the chess board.

    Returns:
    - List[str]: List of legal moves in SAN format.
    """
    return [board.san(move) for move in board.legal_moves]


def get_color(board_turn: bool) -> str:
    """
    Get the color of the current player.

    Parameters:
    - board_turn (bool): Boolean indicating the current player.

    Returns:
    - str: "WHITE" if the current player is white, else "BLACK".
    """
    return "WHITE" if chess.WHITE == board_turn else "BLACK"


def get_legal_move_evaluations(board: chess.Board,
                               engine: chess.engine.SimpleEngine,
                               plot_evaluations: bool = False) \
                                -> Optional[Dict[str, Any]]:
    """
    Evaluate all legal moves from the current board state.

    Parameters:
    - board (chess.Board): The current state of the chess board.
    - engine (chess.engine.SimpleEngine): The chess engine for evaluation.
    - plot_evaluations (bool): Whether to plot the evaluations.
    Default is False.

    Returns:
    - Optional[Dict[str, Any]]: Dictionary containing evaluation
    results of the moves.
    """
    results: Dict[str, Any] = {}
    player = board.turn

    result = engine.analyse(board, chess.engine.Limit(time=0.1))
    prev_score = result["score"].pov(color=player).score(mate_score=900)
    results["prev_score"] = prev_score

    moves = []
    scores_diff = []
    legal_moves = get_san_legal_moves(board)
    if len(legal_moves) == 0:
        return None

    for move in legal_moves:
        board_copy = board.copy()
        board_copy.push_san(move)

        result = engine.analyse(board_copy, chess.engine.Limit(time=0.05))
        evaluation_score = result["score"].pov(
            color=player).score(mate_score=900)

        moves.append(move)
        scores_diff.append(evaluation_score - prev_score)

    scores_mean = np.mean(scores_diff)
    scores_min = min(scores_diff)
    scores_max = max(scores_diff)

    move_min = moves[np.argmin(scores_diff)]
    move_max = moves[np.argmax(scores_diff)]

    results["best_move"] = move_max
    results["worst_move"] = move_min
    results["min_score"] = scores_min
    results["max_score"] = scores_max
    results["mean_score"] = scores_mean

    sorted_moves_evaluations = sorted(
        zip(moves, scores_diff), key=lambda x: x[1], reverse=True)
    sorted_moves, sorted_evaluations = zip(*sorted_moves_evaluations)

    results["sorted_moves"] = list(sorted_moves)
    results["sorted_diff_scores"] = list(sorted_evaluations)

    if plot_evaluations:
        plt.figure(figsize=(14, 5))
        plt.bar(sorted_moves, sorted_evaluations)
        plt.xlabel('Moves')
        plt.ylabel('Score difference')
        plt.title('Sorted Evaluation Differences and Corresponding Moves')
        plt.xticks(rotation=45)
        plt.show()

    return results


def plot_results_hist(
        scores: List[float],
        min_score: float,
        max_score: float,
        title: str = "") -> None:
    """
    Plot a histogram of score differences.

    Parameters:
    - scores (List[float]): List of score differences.
    - min_score (float): Minimum score for histogram bins.
    - max_score (float): Maximum score for histogram bins.
    - title (str): Title of the histogram. Default is an empty string.
    """

    num_bins = 10
    bin_edges = np.linspace(min_score, max_score, num_bins + 1)

    hist, bins = np.histogram(scores, bins=bin_edges)

    plt.figure(figsize=(5, 4))
    plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge')
    plt.xlabel('Score difference')
    plt.ylabel('Frequency')
    plt.title(f'Histogram {title}')
    plt.grid(axis='y')
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

    plt.yticks(np.arange(0, max(hist) + 1, 1))
    plt.show()


def eval_single_position_bot(board: chess.Board,
                             moves: str,
                             move_stats: Dict[str,
                                              Any],
                             engine: chess.engine.SimpleEngine,
                             generator: TextGenerationPipeline,
                             tokenizer: AutoTokenizer,
                             num_answers: int = 100,
                             verbose: bool = False) -> Dict[str,
                                                            Any]:
    """
    Evaluates multiple answers for a single position using the GPT-2 model.

    Parameters:
    - board (chess.Board): The current state of the chess board
    generated with get_state_after_moves().
    - moves (str): Corresponding moves generated with get_state_after_moves().
    - move_stats (Dict[str, Any]): chess engine (Stockfish) move evaluations
    created with get_legal_move_evaluations().
    - engine (chess.engine.SimpleEngine): chess engine (Stockfish).
    - generator (TextGenerationPipeline): The GPT-2 text generation pipeline.
    - tokenizer (AutoTokenizer): Tokenizer for the GPT-2 model.
    - num_answers (int): Number of evaluated answers. Default is 100.
    - verbose (bool): Print evaluation info. Default is False.

    Returns:
    - Dict[str, Any]: Performance statistics including the number of illegal
    moves, legal moves, above-average moves, worst moves, best moves, and score
    differences.
    """

    performances: Dict[str, Any] = {
        "illegal_moves": 0,
        "legal_moves": 0,
        "above_average_moves": 0,
        "worst_moves": 0,
        "best_moves": 0,
        "diff_scores": None
    }
    player = board.turn

    scores = []
    answers = generate_next_moves(
        moves,
        generator,
        tokenizer,
        answer_length=10,
        num_answers=num_answers)
    legal_moves = get_san_legal_moves(board)
    for answer in answers:
        try:
            next_moves = [
                move for move in answer.split(" ") if move[0].isalpha()]
        except IndexError:  # Assume fail if this fails - no clear move
            next_moves = []

        if len(next_moves) == 0:
            if verbose:
                print("No valid moves")
            performances["illegal_moves"] += 1
            continue

        nextmove = next_moves[0]
        if verbose:
            print(
                f"\nGPT Player: {get_color(player)} | Move(?):"
                f" {nextmove} | In legal moves: {nextmove in legal_moves}"
            )

        try:
            board_copy = board.copy()
            board_copy.push_san(nextmove)
            result = engine.analyse(board_copy, chess.engine.Limit(time=0.05))
            evaluation_score = result["score"].pov(
                color=player).score(mate_score=900)
            evaluation_diff = evaluation_score - move_stats['prev_score']

            performances["legal_moves"] += 1

            if verbose:
                print(
                    f" - Score before move: {move_stats['prev_score']}\n"
                    f" - Score after move: {evaluation_score}\n"
                    f" - Is best move: {nextmove==move_stats['best_move']}\n"
                    f" - Is worst move: {nextmove==move_stats['worst_move']}\n"
                    f" - Is above average move: "
                    f"{evaluation_diff > move_stats['mean_score']}"
                )

            scores.append(evaluation_diff)
            if nextmove == move_stats['best_move']:
                performances["best_moves"] += 1

            if nextmove == move_stats['worst_move']:
                performances["worst_moves"] += 1

            if evaluation_diff > move_stats['mean_score']:
                performances["above_average_moves"] += 1

        except (IllMoveError, InvalidMoveError, AmbiguousMoveError):
            performances["illegal_moves"] += 1

    performances["diff_scores"] = scores
    return performances


def eval_single_position_random(board: chess.Board,
                                move_stats: Dict[str,
                                                 Any],
                                engine: chess.engine.SimpleEngine,
                                num_answers: int = 100,
                                verbose: bool = False) -> Dict[str,
                                                               Any]:
    """
    Evaluates multiple answers for a single position using a random player.

    Parameters:
    - board (chess.Board): The current state of the chess
    board generated with get_state_after_moves().
    - move_stats (Dict[str, Any]): chess engine (Stockfish) move evaluations
    created with get_legal_move_evaluations().
    - engine (chess.engine.SimpleEngine): chess engine (Stockfish).
    - num_answers (int): Number of evaluated answers. Default is 100.
    - verbose (bool): Print evaluation info. Default is False.

    Returns:
    - Dict[str, Any]: Performance statistics including the number of
    above-average moves, worst moves, best moves, and score differences.
    """
    performances_random: Dict[str, Any] = {
        "above_average_moves": 0,
        "worst_moves": 0,
        "best_moves": 0,
        "diff_scores": None
    }
    player = board.turn

    scores = []
    legal_moves = get_san_legal_moves(board)
    answers = random.choices(legal_moves, k=num_answers)
    for next_move in answers:
        if verbose:
            print(
                f"\\Random Player: {get_color(player)} | Move(?): {next_move}")

        board_copy = board.copy()
        board_copy.push_san(next_move)
        result = engine.analyse(board_copy, chess.engine.Limit(time=0.05))
        evaluation_score = result["score"].pov(
            color=player).score(mate_score=900)
        evaluation_diff = evaluation_score - move_stats['prev_score']

        if verbose:
            print(
                f" - Score before move: {move_stats['prev_score']}\n"
                f" - Score after move: {evaluation_score}\n"
                f" - Is best move: {next_move==move_stats['best_move']}\n"
                f" - Is worst move: {next_move==move_stats['worst_move']}\n"
                f" - Is above average move:"
                f" {evaluation_diff>move_stats['mean_score']}"
            )

        scores.append(evaluation_diff)
        if next_move == move_stats['best_move']:
            performances_random["best_moves"] += 1

        if next_move == move_stats['worst_move']:
            performances_random["worst_moves"] += 1

        if evaluation_diff > move_stats['mean_score']:
            performances_random["above_average_moves"] += 1

    performances_random["diff_scores"] = scores
    return performances_random


def run_single_game_eval(transcripts: List[str],
                         num_moves: int,
                         num_answers: int,
                         engine: chess.engine.SimpleEngine,
                         generator: TextGenerationPipeline,
                         tokenizer: AutoTokenizer,
                         print_conclusions: bool = False,
                         show_plots: bool = False) -> Tuple[Dict[str,
                                                                 Any],
                                                            Dict[str,
                                                                 Any],
                                                            Dict[str,
                                                                 Any]]:
    """
    Runs a test for single game move analysis.

    Parameters:
    - transcripts (List[str]): Transcripts of games to be used.
    - num_moves (int): Number of moves from the games to be
    played before evaluation move.
    - num_answers (int): Number of model answers generated.
    - engine (chess.engine.SimpleEngine): chess engine (Stockfish).
    - generator (TextGenerationPipeline): Pipeline generator.
    - tokenizer (AutoTokenizer): Model's tokenizer.
    - print_conclusions (bool): Display conclusions. Default is False.
    - show_plots (bool): Display plots. Default is False.

    Returns:
    - Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]: Performance
    statistics for GPT-2, random, and player moves including the number
    of best moves, worst moves, above-average moves, and score differences.
    """
    avg_scores_gpt = []
    avg_scores_random = []
    player_scores = []
    total_performance_gpt: Dict[str, Any] = {
        "total_best_moves": 0,
        "total_worst_moves": 0,
        "total_above_average_moves": 0,
        "total_illegal_moves": 0,
        "total_legal_moves": 0,
        "avg_diff_scores": None
    }
    total_performance_random: Dict[str, Any] = {
        "total_best_moves": 0,
        "total_worst_moves": 0,
        "total_above_average_moves": 0,
        "avg_diff_scores": None
    }
    total_performance_player: Dict[str, Any] = {
        "total_best_moves": 0,
        "total_worst_moves": 0,
        "total_above_average_moves": 0,
        "diff_scores": None
    }

    for i, transcript in enumerate(transcripts):
        if print_conclusions:
            print(
                "=================================="
                "=============================================")
            print(
                f"Game {i} | Number of moves: {num_moves} "
                f"| Evaluated answers: {num_answers}")
            print(
                "==================================="
                "============================================")
        # Get the board and move string after num_moves of the transcripted
        # game
        board, moves, player_move = get_state_after_moves(
            transcript, num_moves)
        if (board is None) or (moves is None):
            print(f"{num_moves} is too many moves for game {i}, skipping...")
            continue

        if show_plots:
            show_board(board, 300)

        # Get the stats of possible legal moves
        move_stats = get_legal_move_evaluations(
            board, engine, plot_evaluations=show_plots)
        if move_stats is None:
            continue

        if print_conclusions:
            print(
                f"Worst move: {move_stats['worst_move']}"
                f" score difference: {move_stats['min_score']}\n"
                f"Best move: {move_stats['best_move']}"
                f" score difference: {move_stats['max_score']}\n" +
                f"Mean difference score: {move_stats['mean_score']}"
            )

        # Get stats for gpt model
        performances = eval_single_position_bot(
            board,
            moves,
            move_stats,
            engine,
            generator,
            tokenizer,
            num_answers=num_answers)
        if print_conclusions:
            print(
                "----------------------------------------------"
                "-------------------------------------\n"
                f"GPT-2 results:\n"
                f"  - Legal moves made: {performances['legal_moves']}"
                f" / {num_answers}\n"
                f"  - Number of times the worst move was made:"
                f" {performances['worst_moves']}\n"
                f"  - Number of times the best move was made:"
                f" {performances['best_moves']}\n"
                f"  - Number of times an above average move "
                f"was made: {performances['above_average_moves']}"
            )

        if performances["legal_moves"] > 0:
            avg_scores_gpt.append(np.mean(performances["diff_scores"]))

        total_performance_gpt["total_above_average_moves"] += \
            performances["above_average_moves"]
        total_performance_gpt["total_best_moves"] += \
            performances["best_moves"]
        total_performance_gpt["total_worst_moves"] += \
            performances["worst_moves"]
        total_performance_gpt["total_illegal_moves"] += \
            performances["illegal_moves"]
        total_performance_gpt["total_legal_moves"] += \
            performances["legal_moves"]

        if show_plots:
            plot_results_hist(
                performances["diff_scores"],
                move_stats["min_score"],
                move_stats["max_score"],
                "Chess-GPT")

        # Get stats for random choice (num repetitions equal
        # to num legal moves made by gpt model)
        if performances["legal_moves"] > 0:
            performances_random = eval_single_position_random(
                board, move_stats, engine, performances["legal_moves"])
            if print_conclusions:
                print(
                    "--------------------------------------------------"
                    "---------------------------------\n"
                    f"Random results:\n"
                    f"  - Number of times the worst move was made: "
                    f"{performances_random['worst_moves']}\n"
                    f"  - Number of times the best move was made: "
                    f"{performances_random['best_moves']}\n"
                    f"  - Number of times an above average move was made:"
                    f" {performances_random['above_average_moves']}"
                )

            avg_scores_random.append(
                np.mean(performances_random["diff_scores"]))
            total_performance_random["total_above_average_moves"] \
                += performances_random["above_average_moves"]
            total_performance_random["total_best_moves"] += \
                performances_random["best_moves"]
            total_performance_random["total_worst_moves"] += \
                performances_random["worst_moves"]

            if show_plots:
                plot_results_hist(
                    performances_random["diff_scores"],
                    move_stats["min_score"],
                    move_stats["max_score"],
                    "Random")

            if player_move is not None:
                board_copy = board.copy()
                board_copy.push_san(player_move)
                result = engine.analyse(
                    board_copy, chess.engine.Limit(
                        time=0.05))
                evaluation_score = result["score"].pov(
                    color=board.turn).score(mate_score=900)
                evaluation_diff = evaluation_score - move_stats['prev_score']

                if evaluation_diff > move_stats["mean_score"]:
                    total_performance_player["total_above_average_moves"] += 1

                if player_move == move_stats["best_move"]:
                    total_performance_player["total_best_moves"] += 1

                if player_move == move_stats["worst_move"]:
                    total_performance_player["total_worst_moves"] += 1

                player_scores.append(evaluation_diff)
                if print_conclusions:
                    print(
                        "---------------------------------------------"
                        "--------------------------------------"
                    )
                    print(f"Player move score difference: {evaluation_diff}")

        elif print_conclusions:
            print(
                '-------------------------------------'
                '----------------------------------------------\n'
                'Random results not generated, no legal moves made by the bot'
            )
    total_performance_player["diff_scores"] = player_scores
    total_performance_gpt["avg_diff_scores"] = avg_scores_gpt
    total_performance_random["avg_diff_scores"] = avg_scores_random

    return total_performance_gpt, total_performance_random, \
        total_performance_player


def show_results_single_eval(summaries: Dict[int,
                                             Dict[str,
                                                  Dict[str,
                                                       Any]]],
                             tests_num_moves: List[int],
                             plot_hist: bool = True) -> None:
    """
    Displays the results of single-game evaluation.

    Parameters:
    - summaries (Dict[int, Dict[str, Dict[str, Any]]]): Summarized results from
    single-game evaluation.
    - tests_num_moves (List[int]): List of numbers of moves.
    - plot_hist (bool): Whether to plot histograms. Default is True.
    """
    aggregate_summary_bot = {
        "agg_legal": 0,
        "agg_illegal": 0,
        "agg_best": 0,
        "agg_worst": 0,
        "agg_above_average": 0
    }

    aggregate_summary_random = {
        "agg_best": 0,
        "agg_worst": 0,
        "agg_above_average": 0
    }

    aggregate_summary_player = {
        "agg_best": 0,
        "agg_worst": 0,
        "agg_above_average": 0
    }

    plot_hist = True

    avg_scores_bot = []
    avg_scores_random = []
    avg_scores_player = []
    for num_moves in tests_num_moves:
        print(
            "============================================"
            "============================"
        )
        print(f"Number of moves before evaluation: {num_moves}")
        print("---------------------------------------\nGPT-2:")
        avg_bot = summaries[num_moves]["gpt"]["avg_diff_scores"]
        avg_score_bot = np.mean(avg_bot)
        avg_scores_bot.append(avg_score_bot)

        total_leg_moves = summaries[num_moves]["gpt"]["total_legal_moves"] + \
            summaries[num_moves]["gpt"]["total_illegal_moves"]

        print(f' - Average score difference after move: {avg_score_bot}')
        print(
            f' - Total number of legal moves: '
            f'{summaries[num_moves]["gpt"]["total_legal_moves"]}'
            f' / {total_leg_moves}'
        )
        print(
            f' - Total number of best moves: '
            f'{summaries[num_moves]["gpt"]["total_best_moves"]}'
            ' / {summaries[num_moves]["gpt"]["total_legal_moves"]}'
        )
        print(
            f' - Total number of worst moves: '
            f'{summaries[num_moves]["gpt"]["total_worst_moves"]}'
            f' / {summaries[num_moves]["gpt"]["total_legal_moves"]}'
        )
        print(
            f' - Total number of above average moves: '
            f'{summaries[num_moves]["gpt"]["total_above_average_moves"]}'
            f' / {summaries[num_moves]["gpt"]["total_legal_moves"]}'
        )

        if plot_hist:
            plot_results_hist(
                avg_bot,
                min(avg_bot),
                max(avg_bot),
                f"Chess GPT {num_moves} moves average score difference")

        print("---------------------------------------\nRandom:")
        avg_random = summaries[num_moves]["random"]["avg_diff_scores"]
        avg_score_random = np.mean(avg_random)
        avg_scores_random.append(avg_score_random)
        print(f' - Average score difference after move: {avg_score_random}')
        print(
            f' - Total number of best moves: '
            f'{summaries[num_moves]["random"]["total_best_moves"]}'
            f' / {summaries[num_moves]["gpt"]["total_legal_moves"]}'
        )
        print(
            f' - Total number of worst moves: '
            f'{summaries[num_moves]["random"]["total_worst_moves"]}'
            f' / {summaries[num_moves]["gpt"]["total_legal_moves"]}'
        )
        print(
            f' - Total number of above average moves: '
            f'{summaries[num_moves]["random"]["total_above_average_moves"]}'
            f' / {summaries[num_moves]["gpt"]["total_legal_moves"]}'
        )

        if plot_hist:
            plot_results_hist(
                avg_random,
                min(avg_random),
                max(avg_random),
                f"Random {num_moves} moves average score difference")

        print("---------------------------------------\nPlayer:")
        avg_player = summaries[num_moves]["player"]["diff_scores"]
        avg_score_player = np.mean(avg_player)
        avg_scores_player.append(avg_score_player)
        print(f' - Average score difference after move: {avg_score_player}')
        print(
            f' - Total number of best moves: '
            f'{summaries[num_moves]["player"]["total_best_moves"]}'
            f' / {len(summaries[num_moves]["player"]["diff_scores"])}'
        )
        print(
            f' - Total number of worst moves: '
            f'{summaries[num_moves]["player"]["total_worst_moves"]}'
            f' / {len(summaries[num_moves]["player"]["diff_scores"])}'
        )
        print(
            f' - Total number of above average moves: '
            f'{summaries[num_moves]["player"]["total_above_average_moves"]}'
            f' / {len(summaries[num_moves]["player"]["diff_scores"])}')

        aggregate_summary_bot["agg_legal"] += \
            summaries[num_moves]["gpt"]["total_legal_moves"]
        aggregate_summary_bot["agg_illegal"] += \
            summaries[num_moves]["gpt"]["total_illegal_moves"]
        aggregate_summary_bot["agg_best"] += \
            summaries[num_moves]["gpt"]["total_best_moves"]
        aggregate_summary_bot["agg_worst"] += \
            summaries[num_moves]["gpt"]["total_worst_moves"]
        aggregate_summary_bot["agg_above_average"] += \
            summaries[num_moves]["gpt"]["total_above_average_moves"]

        aggregate_summary_random["agg_best"] += \
            summaries[num_moves]["random"]["total_best_moves"]
        aggregate_summary_random["agg_worst"] += \
            summaries[num_moves]["random"]["total_worst_moves"]
        aggregate_summary_random["agg_above_average"] += \
            summaries[num_moves]["random"]["total_above_average_moves"]

        aggregate_summary_player["agg_best"] += \
            summaries[num_moves]["player"]["total_best_moves"]
        aggregate_summary_player["agg_worst"] += \
            summaries[num_moves]["player"]["total_worst_moves"]
        aggregate_summary_player["agg_above_average"] += \
            summaries[num_moves]["player"]["total_above_average_moves"]

    gpt_legal_total = aggregate_summary_bot["agg_illegal"] + \
        aggregate_summary_bot["agg_legal"]
    print("\n===============================================================")
    print("Aggregate results GPT-2")
    print(
        f' - Average score difference after move: '
        f'{np.mean(avg_scores_bot)}'
    )
    print(
        f' - Total number of legal moves: '
        f'{aggregate_summary_bot["agg_legal"]} / '
        f'{gpt_legal_total}'
    )
    print(
        f' - Total number of best moves: '
        f'{aggregate_summary_bot["agg_best"]} / '
        f'{aggregate_summary_bot["agg_legal"]}'
    )
    print(
        f' - Total number of worst moves: '
        f'{aggregate_summary_bot["agg_worst"]} / '
        f'{aggregate_summary_bot["agg_legal"]}'
    )
    print(
        f' - Total number of above average moves: '
        f'{aggregate_summary_bot["agg_above_average"]} / '
        f'{aggregate_summary_bot["agg_legal"]}'
    )

    print(
        "---------------------------"
        "-------------------------------------"
    )
    print("Aggregate results Random")
    print(
        f' - Average score difference after move: '
        f'{np.mean(avg_scores_random)}'
    )
    print(
        f' - Total number of best moves: '
        f'{aggregate_summary_random["agg_best"]} / '
        f'{aggregate_summary_bot["agg_legal"]}'
    )
    print(
        f' - Total number of worst moves: '
        f'{aggregate_summary_random["agg_worst"]} / '
        f'{aggregate_summary_bot["agg_legal"]}'
    )
    print(
        f' - Total number of above average moves: '
        f'{aggregate_summary_random["agg_above_average"]} / '
        f'{aggregate_summary_bot["agg_legal"]}'
    )

    print(
        "---------------------------"
        "-------------------------------------"
    )
    print("Aggregate results Player")
    n_moves_player = sum([len(summaries[x]["player"]["diff_scores"])
                         for x in tests_num_moves])
    print(
        f' - Average score difference after move: '
        f'{np.mean(avg_scores_player)}'
    )
    print(
        f' - Total number of best moves: '
        f'{aggregate_summary_player["agg_best"]} / '
        f'{n_moves_player}'
    )
    print(
        f' - Total number of worst moves: '
        f'{aggregate_summary_player["agg_worst"]} / '
        f'{n_moves_player}'
    )
    print(
        f' - Total number of above average moves: '
        f'{aggregate_summary_player["agg_above_average"]} / '
        f'{n_moves_player}'
    )