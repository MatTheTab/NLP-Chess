# Data Analysis

## Contents
This directory includes data related analysis and data reading, it shows how we read the data from the PGN files to a csv format suitable for feeding it into a model or analysing it.

## How to Run

1. Make sure you file structure resembles the one described in README.md of the repository, in particular you should have directories for StockFish, Data and Preprocessed_Data (this one can be empty for now)
2. Make sure your Data directory contains appropriate files, which can be downloaded from the lichess website (Train_Data, Val_Data, Test_Data)
3. Download packages specified in the requirements.txt file
4. Run the data_reading.ipynb file first (can run for many hours, but you can decrease number of games in the notebook)
5. Run the EDA.ipynb file after the data_reading.ipynb stops running
6. That is all -> Thank you!