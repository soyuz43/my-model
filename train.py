# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.model import LanguageModel
from utils.data_loader import get_data_loader
from utils.vocabulary import Vocabulary
import pickle
import os

# --------------------------
# 1. Hyperparameters
# --------------------------
SEQUENCE_LENGTH = 5
BATCH_SIZE = 128
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
CHECKPOINT_DIR = "./checkpoints"
VOCAB_PATH = "./data/vocab.pkl"  # Path to load the vocabulary
PREPROCESSED_DATA_PATH = "./data/preprocessed_data.pkl"  # Path to load the numericalized data

# Updated Attention Parameters
NUM_HEADS = 16  # Must divide HIDDEN_DIM evenly
HEAD_DIVISION = {
    'ideational': 5,
    'interpersonal': 5,
    'textual': 6
}  # Sum must equal NUM_HEADS

# --------------------------
# 2. Prepare Environment
# --------------------------
# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --------------------------
# 3. Load Vocabulary
# --------------------------
print("Loading the vocabulary...")
vocab = Vocabulary.load_vocab(VOCAB_PATH)
print(f"Vocabulary loaded. Size: {len(vocab.token_to_idx)}")

# --------------------------
# 4. Load Numericalized Data
# --------------------------
print("Loading numericalized data...")
with open(PREPROCESSED_DATA_PATH, "rb") as f:
    tokenized_data = pickle.load(f)  # This is a list of token indices
print(f"Numericalized data loaded. Total tokens: {len(tokenized_data)}")

# --------------------------
# 5. Initialize DataLoader
# --------------------------
print("Initializing DataLoader...")
loader = get_data_loader(tokenized_data, SEQUENCE_LENGTH, BATCH_SIZE)
print(f"Loaded dataset size: {len(loader.dataset)} sequences")
print(f"Vocabulary size: {len(vocab.token_to_idx)}")

# --------------------------
# 6. Initialize the Model
# --------------------------
print("Initializing the model...")
model = LanguageModel(
    vocab_size=len(vocab.token_to_idx),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    num_heads=NUM_HEADS,
    head_division=HEAD_DIVISION
)
print("Model initialized.")

# --------------------------
# 7. Define Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# --------------------------
# 8. Define Optimizer and Loss Function
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------
# 9. Initialize TensorBoard Writer
# --------------------------
writer = SummaryWriter(log_dir="./runs/language_model")

# --------------------------
# 10. Log Embeddings for Visualization
# --------------------------
# Log embeddings at the start (epoch=0) and after each epoch
def log_embeddings(model, writer, vocab, epoch):
    # Select a subset of embeddings to visualize (e.g., first 1000 tokens)
    num_embeddings = min(1000, len(vocab.token_to_idx))
    embeddings = model.embedding.weight[:num_embeddings].detach().cpu()
    metadata = [vocab.lookup_token(i) for i in range(num_embeddings)]
    
    writer.add_embedding(
        embeddings,
        metadata=metadata,
        label_img=None,
        global_step=epoch,
        tag='word_embeddings'
    )

# Log the initial embeddings before training
log_embeddings(model, writer, vocab, epoch=0)

# --------------------------
# 11. Training Loop
# --------------------------
print("Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    epoch_loss = 0

    # Create progress bar
    progress_bar = tqdm(loader, desc="Training Progress", leave=False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Validate inputs
        if inputs.max() >= len(vocab.token_to_idx) or inputs.min() < 0:
            raise ValueError(f"Invalid token indices in inputs: {inputs}")

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(inputs)  # Shape: (batch_size, vocab_size)

        # Compute loss
        loss = criterion(outputs, targets)  # targets shape: (batch_size,)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update epoch loss
        epoch_loss += loss.item()

        # Log batch loss
        global_step = (epoch - 1) * len(loader) + batch_idx
        writer.add_scalar("Loss/Batch", loss.item(), global_step)

        # Log gradients and weights histograms periodically (e.g., every 100 batches)
        if batch_idx % 100 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"Gradients/{name}", param.grad, global_step)
                writer.add_histogram(f"Weights/{name}", param.data, global_step)

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(loader)
    writer.add_scalar("Loss/Epoch", avg_loss, epoch)

    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

    # Save the model checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

    # Log embeddings after each epoch
    log_embeddings(model, writer, vocab, epoch)

# Close the TensorBoard writer
writer.close()

# --------------------------
# 12. Sanity Check After Training
# --------------------------
print("\nTraining complete. Running a sanity check on the model...")

# Function to generate the next word (token index)
def generate_next_token(model, input_seq, device):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)  # Shape: (1, sequence_length)
        outputs, _ = model(input_tensor)  # Shape: (1, vocab_size)
        predicted_idx = outputs.argmax(dim=-1).item()
        return predicted_idx

# Example input sequence (first SEQUENCE_LENGTH tokens from the dataset)
example_sequence = numericalized_data[:SEQUENCE_LENGTH]
print(f"Input Sequence (token indices): {example_sequence}")
predicted_token = generate_next_token(model, example_sequence, device)
print(f"Predicted Next Token Index: {predicted_token}")

# Optional: Convert predicted token index back to token
predicted_word = vocab.lookup_token(predicted_token)
print(f"Predicted Next Token: {predicted_word}")
