# generate.py

import torch
import torch.nn as nn
import argparse
from models.model import LanguageModel
from utils.vocabulary import Vocabulary
import pickle
import os

def load_vocabulary(vocab_path):
    """
    Loads the vocabulary from a pickle file.
    
    Args:
        vocab_path (str): Path to the vocab.pkl file.
    
    Returns:
        Vocabulary: An instance of the Vocabulary class.
    """
    vocab = Vocabulary.load_vocab(vocab_path)
    return vocab

def load_model(checkpoint_path, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, head_division, dropout=0.2, max_seq_len=5000):
    """
    Initializes the model architecture and loads the trained weights.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth file).
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of token embeddings.
        hidden_dim (int): Dimension of hidden layers.
        num_layers (int): Number of Transformer encoder layers.
        num_heads (int): Number of attention heads.
        head_division (dict): Division of heads among metafunctions.
        dropout (float, optional): Dropout rate. Default is 0.2.
        max_seq_len (int, optional): Maximum sequence length for positional encoding. Default is 5000.
    
    Returns:
        LanguageModel: The initialized model with loaded weights.
    """
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
    model.eval()  # Set model to evaluation mode
    
    return model

def tokenize(prompt):
    """
    Tokenizes the input prompt. This function should mirror the tokenization process used during training.
    Modify this function based on your actual tokenization method.
    
    Args:
        prompt (str): The input text prompt.
    
    Returns:
        List[str]: A list of tokens.
    """
    # Example tokenization: simple whitespace tokenizer. Replace with your actual tokenizer.
    tokens = prompt.strip().split()
    return tokens

def generate_text(model, vocab, prompt, max_length=50, temperature=1.0, top_k=0, top_p=0.0):
    """
    Generates text based on the input prompt using the trained model.
    
    Args:
        model (LanguageModel): The trained language model.
        vocab (Vocabulary): The vocabulary instance.
        prompt (str): The input text prompt.
        max_length (int, optional): Maximum number of tokens to generate. Default is 50.
        temperature (float, optional): Sampling temperature. Higher values increase randomness.
        top_k (int, optional): Limits sampling to the top_k most probable tokens. 0 disables.
        top_p (float, optional): Limits sampling to a cumulative probability of top_p. 0.0 disables.
    
    Returns:
        str: The generated text.
    """
    model.eval()
    
    # Tokenize and numericalize the prompt
    tokens = tokenize(prompt)
    numericalized = [vocab.lookup_index(token) for token in tokens]
    
    # Convert to tensor
    input_tensor = torch.tensor([numericalized], dtype=torch.long)  # Shape: (1, sequence_length)
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model outputs
            outputs, _ = model(input_tensor)  # Shape: (1, vocab_size)
            logits = outputs.squeeze(0)  # Shape: (vocab_size,)
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top_k and/or top_p filtering
            if top_k > 0:
                # Keep only top_k tokens with highest probability
                top_k = min(top_k, len(logits))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter the removal mask to the original indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample the next token
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if EOS token is generated (if defined)
            eos_token = '<eos>'
            if eos_token in vocab.token_to_idx:
                if next_token_idx == vocab.lookup_index(eos_token):
                    break
            
            # Append the predicted token to the generated list
            next_token = vocab.lookup_token(next_token_idx)
            generated.append(next_token)
            
            # Prepare the input for the next iteration
            input_tensor = torch.tensor([numericalized + [next_token_idx]], dtype=torch.long)
    
    # Join the tokens into a single string
    generated_text = ' '.join(generated)
    return generated_text

def main():
    # --------------------------
    # 1. Argument Parsing
    # --------------------------
    parser = argparse.ArgumentParser(description="Language Model Text Generation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--vocab', type=str, required=True, help='Path to the vocabulary file (vocab.pkl)')
    parser.add_argument('--prompt', type=str, required=False, default='', help='Input text prompt for generation')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top-K sampling (0 to disable)')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top-P (nucleus) sampling (0.0 to disable)')
    args = parser.parse_args()
    
    # --------------------------
    # 2. Load Vocabulary
    # --------------------------
    print("Loading vocabulary...")
    vocab = load_vocabulary(args.vocab)
    print(f"Vocabulary size: {len(vocab.token_to_idx)}")
    
    # --------------------------
    # 3. Load Model
    # --------------------------
    print("Loading model...")
    # Retrieve model parameters from the checkpoint or define them manually
    # Here, we'll assume you know the model's hyperparameters
    # Alternatively, you can save hyperparameters in the checkpoint and load them dynamically
    # For simplicity, define them manually based on your training script
    
    # Define the same hyperparameters used during training
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    NUM_HEADS = 16
    HEAD_DIVISION = {
        'ideational': 5,
        'interpersonal': 5,
        'textual': 6
    }
    DROPOUT = 0.2
    MAX_SEQ_LEN = 5000  # Adjust if different during training
    
    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab_size=len(vocab.token_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_division=HEAD_DIVISION,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    )
    print("Model loaded successfully.")
    
    # --------------------------
    # 4. Interactive Generation
    # --------------------------
    while True:
        prompt = input("\nEnter a prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Exiting text generation.")
            break
        elif prompt.strip() == '':
            print("Please enter a valid prompt.")
            continue
        
        generated_text = generate_text(
            model=model,
            vocab=vocab,
            prompt=prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        print("\nGenerated Text:\n" + generated_text)

if __name__ == "__main__":
    main()
