# hangman-player

A encoder-only Transformer model trained to play the Hangman word-guessing game. Predicts masked characters in partially revealed words using self-attention mechanisms. The trained model is used to create two different HangmanPlayers.

Players
- HangmanPlayer: Basic player using greedy decoding
- HangmanPlayer2: Advanced player exploring multiple candidate guesses via beam search

Game Simulation
- HangmanServer: Manages game states, tracks guesses, and evaluates success rate

Bonus: Also a LSTM based hangman player trained using the PPO algorithm. Unfortunately, it doesn't work very well. 

(A lot of code is borrowed from Andrej Karpathy's Let's Build GPT tutorial)

## Features
- **Transformer Architecture**: Implements self-attention and positional embeddings
- **Masked Prediction**: Learns to fill in missing characters in words
- **Game Integration**: Includes AI players for Hangman gameplay
- **Beam Search**: Advanced guessing with multiple candidate explorations
- **Training Pipeline**: Customizable hyperparameters for model optimization

## Requirements
- Python 3.8+
- PyTorch 2.0+ (with MPS support recommended for Apple Silicon)
- `words_250000_train.txt` dataset file
- `words_alpha_train_unique.txt` for evaluation
