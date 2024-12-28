# build_vocab.py

import pickle
import os
from utils.vocabulary import Vocabulary

def build_and_save_vocab(preprocessed_data_path, vocab_output_path, freq_threshold=1):
    """
    Builds a vocabulary from the preprocessed token data and saves it to a file.

    Args:
        preprocessed_data_path (str): Path to the preprocessed token data (small-arms-preprocessed.pkl).
        vocab_output_path (str): Path where the vocabulary (vocab.pkl) will be saved.
        freq_threshold (int, optional): Minimum frequency a token must have to be included in the vocabulary.
                                        Default is 1.
    """
    # Load the preprocessed tokens
    print(f"Loading preprocessed data from {preprocessed_data_path}...")
    with open(preprocessed_data_path, "rb") as f:
        tokens = pickle.load(f)  # This should be a list of tokens

    # Since the Vocabulary class expects a list of tokenized sentences (list of lists),
    # we'll treat the entire token list as a single sentence/document.
    tokenized_texts = [tokens]

    # Initialize the Vocabulary
    print("Initializing Vocabulary...")
    vocab = Vocabulary(freq_threshold=freq_threshold)

    # Build the Vocabulary
    print("Building Vocabulary...")
    vocab.build_vocabulary(tokenized_texts)
    print(f"Vocabulary size after building: {len(vocab.token_to_idx)}")

    # Save the Vocabulary
    print(f"Saving Vocabulary to {vocab_output_path}...")
    vocab.save_vocab(vocab_output_path)
    print("Vocabulary successfully saved.")

if __name__ == "__main__":
    # Define paths
    preprocessed_data_path = "./data/merged_preprocessed_data.pkl"
    vocab_output_path = "./data/vocab.pkl"
    
    # Build and save the vocabulary
    build_and_save_vocab(preprocessed_data_path, vocab_output_path, freq_threshold=1)
