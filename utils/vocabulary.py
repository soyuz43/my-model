# utils/vocabulary.py

import pickle
import os
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=1):
        """
        Initializes the Vocabulary.

        Args:
            freq_threshold (int): Minimum frequency a token must have to be included in the vocabulary.
        """
        self.freq_threshold = freq_threshold

        # Initialize token to index and index to token mappings
        self.token_to_idx = {}
        self.idx_to_token = {}

        # Initialize special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token = "<eos>"

        # Add special tokens to the vocabulary
        self.add_special_tokens()

    def add_special_tokens(self):
        """
        Adds special tokens to the vocabulary.
        """
        self.add_token(self.pad_token)
        self.add_token(self.unk_token)
        self.add_token(self.eos_token)

    def add_token(self, token):
        """
        Adds a token to the vocabulary.

        Args:
            token (str): The token to add.
        """
        if not isinstance(token, str):
            raise ValueError(f"Token must be a string, got {type(token)}: {token}")

        if token not in self.token_to_idx:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token

    def build_vocabulary(self, tokenized_texts):
        """
        Builds the vocabulary from a list of tokenized texts.

        Args:
            tokenized_texts (List[List[str]]): A list of tokenized sentences (each sentence is a list of tokens).
        """
        # Flatten the list of tokenized texts
        all_tokens = [token for text in tokenized_texts for token in text]

        # Count token frequencies
        token_freq = Counter(all_tokens)

        # Add tokens to the vocabulary based on frequency threshold
        for token, freq in token_freq.items():
            if freq >= self.freq_threshold:
                self.add_token(token)

    def lookup_index(self, token):
        """
        Returns the index of the given token. If the token is not found, returns the index of <unk>.

        Args:
            token (str): The token to look up.

        Returns:
            int: The index of the token.
        """
        if not isinstance(token, str):
            raise ValueError(f"lookup_index expects a string token, got {type(token)}: {token}")
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])

    def lookup_token(self, index):
        """
        Returns the token corresponding to the given index. If the index is not found, returns <unk>.

        Args:
            index (int): The index to look up.

        Returns:
            str: The corresponding token.
        """
        if not isinstance(index, int):
            raise ValueError(f"lookup_token expects an integer index, got {type(index)}: {index}")
        token = self.idx_to_token.get(index, self.unk_token)
        if not isinstance(token, str):
            # If this ever happens, it indicates data corruption or incorrect vocabulary construction
            raise ValueError(f"lookup_token returned a non-string token: {token} (type: {type(token)})")
        return token

    def save_vocab(self, filepath):
        """
        Saves the vocabulary to a file using pickle.

        Args:
            filepath (str): Path to the file where the vocabulary will be saved.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'token_to_idx': self.token_to_idx,
                'idx_to_token': self.idx_to_token
            }, f)
        print(f"Vocabulary saved to {filepath}")

    @classmethod
    def load_vocab(cls, filepath):
        """
        Loads the vocabulary from a pickle file.

        Args:
            filepath (str): Path to the vocabulary file.

        Returns:
            Vocabulary: An instance of the Vocabulary class.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        vocab = cls()
        vocab.token_to_idx = data['token_to_idx']
        vocab.idx_to_token = data['idx_to_token']

        # Validate that all tokens are strings
        for idx, tok in vocab.idx_to_token.items():
            if not isinstance(tok, str):
                raise ValueError(f"Invalid token at index {idx}: expected a string, got {type(tok)} ({tok})")

        print(f"Vocabulary loaded from {filepath}")
        return vocab
