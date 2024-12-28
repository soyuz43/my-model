import os
import pickle
from datasets import load_dataset
from tqdm import tqdm
from utils.vocabulary import Vocabulary

def tokenize(text):
    # Simple whitespace tokenizer; replace with more sophisticated processing if needed
    return text.split()

# Load the dataset and take a tenth of the training data
dataset = load_dataset("Skylion007/openwebtext")
dataset_size = len(dataset['train'])
tenth_dataset = dataset['train'].select(range(dataset_size // 10))  # Select the first 10% of the dataset

# Load or build the vocabulary
vocab_file = './data/vocab.pkl'
if os.path.exists(vocab_file):
    print("Loading vocabulary...")
    vocab = Vocabulary.load_vocab(vocab_file)
else:
    print("Building vocabulary...")
    vocab = Vocabulary()
    for entry in tqdm(tenth_dataset, desc="Building Vocabulary"):
        tokens = tokenize(entry['text'])
        for token in tokens:
            vocab.add_token(token)
    vocab.save_vocab(vocab_file)
    print(f"Vocabulary built and saved to {vocab_file}")

# Process and save the tokenized data
preprocessed_data_path = './data/preprocessed_data.pkl'
tokenized_data = []
print("Processing text data...")
for entry in tqdm(tenth_dataset, desc="Processing Data"):
    tokens = tokenize(entry['text'])
    numericalized_tokens = [vocab.lookup_index(token) for token in tokens]
    tokenized_data.extend(numericalized_tokens)

with open(preprocessed_data_path, 'wb') as f:
    pickle.dump(tokenized_data, f)
print(f"Data processed and saved to {preprocessed_data_path}")
