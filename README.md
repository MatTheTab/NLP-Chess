# NLP-Chess

Contemporary language models have achieved remarkable results in the tasks of code and text generation, displaying behavior that is often seen as intelligent, logical, and forward-thinking.
As such, it is beneficial to explore and test the abilities of such models in the context of board games requiring strategic abilities and logical thinking to test the current limits of their abilities.
This repository is dedicated to analyzing the current prowess of language models in the domain of game-playing, particularly within the context of chess.

<b> Link to the dataset: </b> https://drive.google.com/drive/folders/1XzzcIMATMWeCUdlQG9BLtsVae5fjphEo?usp=sharing <br>

## Data Explanation

| Column Name          | Description |
|----------------------|-------------|
| game_number          |       Game number, i.e. what is the order of this game starting from 1 when reading the .pgn file. Can be repeated due to many moves making up one game (including fake moves)      |
| move_number          |       The Number of the move, i.e. which move is it in the game starting from 1, can be repeated due to fake moves      |
| board                |       String representation of the board, achieved using the board.fen() method      |
| move                 |       Move representation as a string for example e2e4      |
| legal                |       Bool value, shows if the move was legal      |
| stockfish_2          |       Stockfish evaluation for depth 2 and time limit 0.001 seconds. Can be None if the move was illegal      |
| stockfish_5          |       Stockfish evaluation for depth 5 and time limit 0.001 seconds. Can be None if the move was illegal      |
| stockfish_10         |       Stockfish evaluation for depth 10 and time limit 0.001 seconds. Can be None if the move was illegal      |
| real                 |       Bool value, if the move was made by a real player, or randomly generated      |
| piece_placement      |       Part of the board.fen() representation, string representation of chess positions      |
| active_color         |       Part of the board.fen() representation, string representation of chess positions      |
| castling_availability|       Part of the board.fen() representation, string shows available castling moves      |
| en_passant           |       Part of the board.fen() representation, the string represents En passant target square - if a pawn has just made a two-square move, this is the position "behind" the pawn. If there's no recent pawn move, this is represented as a dash (-)      |
| halfmove_clock       |       Part of the board.fen() representation, string representation shows the number of half-moves since the last capture or pawn move. This is used for the fifty-move rule.      |
| fullmove_number      |       Part of the board.fen() representation, shows the fullmove number - the number of the current full move. It starts at 1 and is incremented after black's move.      |

