# Chess Playing

## Contents
This directory includes notebooks and utilities used for evaluating language model chess playing capabilities.
Please note, evaluate_gpt_2.ipynb can be run locally using packages located in requirements.txt, however
the remaining projects are best run on google colab.

## How to Run

1. Make sure you have the appropriate libraries installed
2. Update the stockfish paths in the notebook you would like to run to the path appropriate for your file system.
3. (Optional) Uncomment and use the load_dataset huggingface function to download the appropriate data if you would like to change the subset of games used.

**Note: Change directory to chess_playing if you intend to run the code as scripts or run tests.**
PS ..\NLP-Chess\chess_playing> pytest .\chessplaying_tests.py