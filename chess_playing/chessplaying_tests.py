import pytest
import torch

import chess
import chess.engine
import chess.svg

from transformers import AutoTokenizer, TextGenerationPipeline, \
    GPT2LMHeadModel, GPT2Tokenizer, pipeline

from datasets import load_from_disk

from typing import List, Dict, Optional, Any

from utils.chessplaying_utils import *

import warnings
warnings.filterwarnings("ignore")


WHITE = 0
BLACK = 1


class TestChessPlaying:
    """
    Test suite for validating chess game
    playing functionality including data loading,
    state retrieval, move generation, and
    forced move generation using GPT-2.
    """
    data_path = "data/subset_games"

    def test_data_loading(self) -> None:
        """
        Test the loading of chess game data and the extraction of moves.
        Ensures that the moves are correctly loaded and legal moves
        are validated.
        """
        data_moves = load_from_disk(self.data_path)
        moves = get_n_moves(data_moves["transcript"][0], 5)  # type: ignore

        assert moves is not None, \
            "Moves should be returned correctly for this game"

        game = [x for x in moves.split(" ") if x[0].isalpha()]

        assert len(game) == 5, \
            "5 moves should be read successfully from the game"

        board = chess.Board()
        for move in game:
            t = "Move should be legal"
            assert move in get_san_legal_moves(board), t  # type: ignore
            board.push_san(move)

    def test_get_state(self) -> None:
        """
        Test the retrieval of the board state after a given number of moves.
        Ensures that the board state, moves, and next player move are
        correctly retrieved and that the moves are legal and correctly
        applied to the board.
        """
        data_moves = load_from_disk(self.data_path)  # type: ignore

        board, moves, p_move = \
            get_state_after_moves(  # type: ignore
                data_moves["transcript"][0], 5)

        assert moves is not None, \
            "Moves should be returned correctly for this game"
        assert board is not None, \
            "Board should be returned correctly for this game"
        assert p_move is not None, \
            "Player move should be returned correctly for this game"

        game = [x for x in moves.split(" ") if x[0].isalpha()]

        assert len(board.move_stack) == 5, \
            "5 moves should be performed and stored in board move stack"
        assert len(game) == len(board.move_stack), \
            "The same number of moves should be played and in the move string"

        t = "Next player move should be legal"
        assert p_move in get_san_legal_moves(board), t  # type: ignore

    def test_generation(self) -> None:
        """
        Test the generation of the next moves using GPT-2.
        Ensures that the text generation pipeline produces
        a non-empty move sequence.
        """
        data_moves = load_from_disk(self.data_path)

        _, moves, _ = get_state_after_moves(  # type: ignore
            data_moves["transcript"][0], 5)

        assert moves is not None, \
            "Moves should be returned correctly for this game"

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        generator = pipeline("text-generation", model="gpt2", device=0)
        generator("Once upon a time,", max_length=40, truncation=True,
                  pad_token_id=generator.tokenizer.eos_token_id)

        answer = generate_next_moves(  # type: ignore
            moves, generator, tokenizer, 20, 1)

        assert len(answer) == 1, \
            "One answer should be produced by the model"
        assert len(answer[0]) > 0, \
            "Answer produced by the model should be nonempty"

    def test_forced_generation(self) -> None:
        """
        Test the generation of text using GPT-2 with
        forced inclusion of specific words.
        Ensures that the generated text includes the specified forced word.
        """
        class AlternativesPipeline(TextGenerationPipeline):
            """
            Custom pipeline for generating text with
            forced inclusion of specified words.
            """
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

            def __call__(self, *args: Any,
                         force_words: Optional[List[str]] = None,
                         **kwargs: Any) -> Any:
                """
                Override the call method to include forced
                words in the text generation.

                Args:
                    force_words (Optional[List[str]]): List of
                    words to force include in the generation.

                Returns:
                    Any: The generated text with forced words.
                """

                force_words_ids = [
                    self.tokenizer(force_words,
                                   add_special_tokens=False).input_ids
                ]

                kwargs["force_words_ids"] = force_words_ids
                return super().__call__(*args, **kwargs)

        t = True
        c = "cuda"
        gpt_h = GPT2LMHeadModel.from_pretrained("gpt2",
                                                no_repeat_ngram_size=1,
                                                remove_invalid_values=t).to(c)

        forced_generator = AlternativesPipeline(
            model=gpt_h,
            tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
            device=0
        )

        alternatives = ["dog"]

        answer = generate_answer(  # type: ignore
            forced_generator,
            "What is your favorite animal? A:",
            max_len=25,
            num_answers=1,
            alternatives=alternatives)

        assert len(answer) == 1, \
            "One answer should be produced by the model"
        assert len(answer[0]) > 0, \
            "Answer produced by the model should be nonempty"
        assert "dog" in answer[0], \
            "Answer contain the forced word"
