import os
import pickle
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import h5py

# Choose a pre-trained tokenizer (e.g., GPT-2)
tokenizer_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Define maximum sequence length
MAX_LENGTH = 1024  # Adjust based on your model's requirements

# Load the dataset and take a subset for faster processing (e.g., 10%)
dataset = load_dataset("Skylion007/openwebtext")
dataset_size = len(dataset['train'])
subset_size = dataset_size // 10  # Using only 10% for speedup
subset = dataset['train'].select(range(subset_size))

# Path to save the HDF5 file
hdf5_path = './data/preprocessed_data.h5'

# Create the HDF5 file
with h5py.File(hdf5_path, 'w') as hdf5_file:
    # Initialize a dataset within the HDF5 file
    # Assuming token IDs are stored as 32-bit integers
    dset = hdf5_file.create_dataset(
        'tokens',
        shape=(0,),
        maxshape=(None,),
        dtype='int32',
        chunks=True  # Enable chunking for efficient resizing
    )

    print("Tokenizing and encoding the dataset...")
    for entry in tqdm(subset, desc="Tokenizing Data"):
        # Tokenize and encode the text
        tokens = tokenizer.encode(entry['text'], truncation=True, max_length=MAX_LENGTH)
        tokens = tokens[:MAX_LENGTH]  # Ensure tokens do not exceed MAX_LENGTH

        # Append tokens to the HDF5 dataset
        current_size = dset.shape[0]
        new_size = current_size + len(tokens)
        dset.resize((new_size,))
        dset[current_size:new_size] = tokens

print(f"Tokenized data saved to {hdf5_path}")
