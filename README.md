# NLP-Chess

Contemporary language models have achieved remarkable results in the tasks of code and text generation, displaying behavior that is often seen as intelligent, logical, and forward-thinking.
As such, it is beneficial to explore and test the abilities of such models in the context of board games requiring strategic abilities and logical thinking to test the current limits of their abilities.
This repository is dedicated to analyzing the current prowess of language models in the domain of game-playing, particularly within the context of chess.

<b> Link to the dataset: </b> https://drive.google.com/drive/folders/1XzzcIMATMWeCUdlQG9BLtsVae5fjphEo?usp=sharing <br>
<b> Paper for BERT retraining: </b> https://arxiv.org/pdf/2209.11902 <br>

## Data Explanation

| Column Name          | Description |
|----------------------|-------------|
| game_number          |       Game number, i.e. what is the order of this game starting from 1 when reading the .pgn file. Can be repeated due to many moves making up one game (including fake moves)      |
| move_number          |       The Number of the move, i.e. which move is it in the game starting from 1, can be repeated due to fake moves      |
| board                |       String representation of the board, achieved using the board.fen() method   after the move was performed   |
| move                 |       Move representation as a string for example e2e4      |
| legal                |       Bool value, shows if the move was legal      |
| stockfish_2          |       Stockfish evaluation for depth 2 for the current board (just "board" from column names) and time limit 0.01 seconds. Can be None if the move was illegal      |
| stockfish_5          |       Stockfish evaluation for depth 5 for the current board (just "board" from column names) and time limit 0.01 seconds. Can be None if the move was illegal      |
| stockfish_10         |       Stockfish evaluation for depth 10 for the current board (just "board" from column names) and time limit 0.01 seconds. Can be None if the move was illegal      |
| move_quality_2       |       Move quality calculated as the difference between earlier board evaluation and the current board evaluation from stockfish for depth 2, can be None for starting moves and if it was illegal    |
| move_quality_5       |       Move quality calculated as the difference between earlier board evaluation and the current board evaluation from stockfish for depth 5, can be None for starting moves and if it was illegal   |
| move_quality_10      |       Move quality calculated as the difference between earlier board evaluation and the current board evaluation from stockfish for depth 10, can be None for starting moves and if it was illegal    |
| prev_ELO             |       ELO of the player performing the move, can be None for random     |
| current_ELO          |       ELO of the player after the move was performed    |
| real                 |       Bool value, if the move was made by a real player, or randomly generated      |
| piece_placement      |       Part of the board.fen() representation, string representation of chess positions      |
| active_color         |       Part of the board.fen() representation, string representation of chess positions shows player color after the move was performed     |
| castling_availability|       Part of the board.fen() representation, string shows available castling moves      |
| en_passant           |       Part of the board.fen() representation, the string represents En passant target square - if a pawn has just made a two-square move, this is the position "behind" the pawn. If there's no recent pawn move, this is represented as a dash (-)      |
| halfmove_clock       |       Part of the board.fen() representation, string representation shows the number of half-moves since the last capture or pawn move. This is used for the fifty-move rule.      |
| fullmove_number      |       Part of the board.fen() representation, shows the fullmove number - the number of the current full move. It starts at 1 and is incremented after black's move.      |
| prev_board           |       String representation of the board before the move was performed by a player, represented using the board.fen() method      |
| prev_piece_placement      |       Part of the board.fen() representation, string representation of chess positions for the previous board, before the move was performed     |
| prev_active_color         |       Part of the board.fen() representation, string representation of chess positions for the previous board, before the move was performed    |
| prev_castling_availability|       Part of the board.fen() representation, string shows available castling moves, for the previous board - before the move was performed      |
| prev_en_passant           |       Part of the board.fen() representation, the string represents En passant target square - if a pawn has just made a two-square move, this is the position "behind" the pawn. If there's no recent pawn move, this is represented as a dash (-), for the previous board - before move was performed      |
| prev_halfmove_clock       |       Part of the board.fen() representation, string representation shows the number of half-moves since the last capture or pawn move. This is used for the fifty-move rule, before the move was performed     |
| prev_fullmove_number      |       Part of the board.fen() representation, shows the fullmove number - the number of the current full move. It starts at 1 and is incremented after black's move, for the previous board - before move was performed      |

