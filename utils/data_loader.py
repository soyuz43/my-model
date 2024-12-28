# utils/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
import nltk

# Optional: Download NLTK data if using NLTK tokenizer
# nltk.download('punkt')
from nltk.tokenize import word_tokenize  # If using NLTK

class LanguageDataset(Dataset):
    def __init__(self, tokenized_data, sequence_length):
        """
        Args:
            tokenized_data (list of int): List of token indices.
            sequence_length (int): Number of tokens in the input sequence.
        """
        self.sequence_length = sequence_length
        self.tokenized_data = tokenized_data
        self.sequences, self.targets = self.create_sequences(tokenized_data, sequence_length)

    def create_sequences(self, data, sequence_length):
        """
        Creates input sequences and corresponding target tokens.

        Args:
            data (list of int): List of token indices.
            sequence_length (int): Number of tokens in the input sequence.

        Returns:
            sequences (list of list of int): List of input sequences.
            targets (list of int): List of target tokens.
        """
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
        return sequences, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

def collate_fn(batch):
    """
    Collate function to stack input sequences and targets.

    Args:
        batch (list of tuples): Each tuple contains (input_tensor, target_tensor).

    Returns:
        inputs (Tensor): Tensor of shape (batch_size, sequence_length).
        targets (Tensor): Tensor of shape (batch_size).
    """
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets

def get_data_loader(tokenized_data, sequence_length, batch_size):
    """
    Creates a DataLoader for the language dataset.

    Args:
        tokenized_data (list of int): List of token indices.
        sequence_length (int): Number of tokens in the input sequence.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader instance.
    """
    dataset = LanguageDataset(tokenized_data, sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader

# Define the tokenize and numericalize functions

def tokenize(text):
    """
    Tokenizes the input text into tokens using NLTK's word_tokenize.

    Args:
        text (str): The input text.

    Returns:
        List[str]: A list of tokens.
    """
    return word_tokenize(text)

def numericalize(tokens, vocab):
    """
    Converts a list of tokens into their corresponding numerical indices using the vocabulary.

    Args:
        tokens (List[str]): List of tokens.
        vocab (Vocabulary): Vocabulary object with token-to-index mapping.

    Returns:
        List[int]: List of token indices.
    """
    return [vocab.lookup_index(token) for token in tokens]
