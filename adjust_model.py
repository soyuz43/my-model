# adjust_model.py

import torch
import torch.nn as nn
from models.model import LanguageModel
from utils.vocabulary import Vocabulary
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Adjust Model for Updated Vocabulary")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the existing model checkpoint (.pth file)')
    parser.add_argument('--updated_vocab', type=str, required=True, help='Path to the updated vocabulary file (updated_vocab.pkl)')
    parser.add_argument('--adjusted_checkpoint', type=str, default='checkpoints/adjusted_model_epoch_15.pth', help='Path to save the adjusted model checkpoint')
    return parser.parse_args()

def load_vocabulary(vocab_path):
    vocab = Vocabulary.load_vocab(vocab_path)
    return vocab

def load_model(checkpoint_path, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, head_division, dropout=0.2, max_seq_len=5000):
    model = LanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        head_division=head_division,
        max_seq_len=max_seq_len
    )
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model

def adjust_model_for_new_vocab(model, original_vocab_size, new_vocab_size, embedding_dim, hidden_dim):
    """
    Adjusts the model's embedding and output layers to accommodate the new vocabulary size.
    
    Args:
        model (LanguageModel): The loaded language model.
        original_vocab_size (int): The original vocabulary size.
        new_vocab_size (int): The updated vocabulary size.
        embedding_dim (int): Dimension of token embeddings.
        hidden_dim (int): Dimension of hidden layers.
    
    Returns:
        LanguageModel: The adjusted model.
    """
    # Adjust the embedding layer
    print("Adjusting the embedding layer...")
    new_embedding = nn.Embedding(new_vocab_size, embedding_dim)
    new_embedding.weight.data[:original_vocab_size] = model.embedding.weight.data
    nn.init.normal_(new_embedding.weight.data[original_vocab_size:], mean=0.0, std=0.02)
    model.embedding = new_embedding
    print("Embedding layer adjusted.")
    
    # Adjust the output (fully connected) layer
    print("Adjusting the output (fc) layer...")
    original_fc = model.fc
    new_fc = nn.Linear(hidden_dim, new_vocab_size)
    new_fc.weight.data[:original_vocab_size] = original_fc.weight.data
    new_fc.bias.data[:original_vocab_size] = original_fc.bias.data
    nn.init.normal_(new_fc.weight.data[original_vocab_size:], mean=0.0, std=0.02)
    model.fc = new_fc
    print("Output layer adjusted.")
    
    return model

def save_adjusted_checkpoint(model, checkpoint_path):
    """
    Saves the adjusted model checkpoint.
    
    Args:
        model (LanguageModel): The adjusted language model.
        checkpoint_path (str): Path to save the adjusted checkpoint.
    """
    torch.save({
        'epoch': 15,  # Update if needed
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},  # Empty if optimizer not used here
        'loss': 0.0  # Placeholder
    }, checkpoint_path)
    print(f"Adjusted model checkpoint saved at {checkpoint_path}")

def main():
    args = parse_args()
    
    # Load updated vocabulary
    print("Loading updated vocabulary...")
    vocab = load_vocabulary(args.updated_vocab)
    new_vocab_size = len(vocab.token_to_idx)
    print(f"Updated vocabulary size: {new_vocab_size}")
    
    # Load existing model
    print("Loading existing model checkpoint...")
    # Define model hyperparameters (must match the original training)
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    NUM_HEADS = 16
    HEAD_DIVISION = {
        'ideational': 5,
        'interpersonal': 5,
        'textual': 6
    }
    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab_size=new_vocab_size,  # Temporarily set to new_vocab_size
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_division=HEAD_DIVISION,
        dropout=0.2,
        max_seq_len=5000
    )
    
    # Determine original vocabulary size from the checkpoint
    original_vocab_size = model.embedding.num_embeddings - (new_vocab_size - len(vocab.token_to_idx))
    print(f"Original vocabulary size: {original_vocab_size}")
    
    # Adjust the model for the new vocabulary size
    model = adjust_model_for_new_vocab(
        model=model,
        original_vocab_size=original_vocab_size,
        new_vocab_size=new_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    # Save the adjusted model checkpoint
    save_adjusted_checkpoint(model, args.adjusted_checkpoint)

if __name__ == "__main__":
    main()
