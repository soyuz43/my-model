from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Skylion007/openwebtext")

# Print dataset information
print("Dataset loaded successfully!")
print(f"Dataset structure: {dataset}")

# Access and print the location of the cache files
dataset_cache_files = dataset['train'].cache_files
if dataset_cache_files:
    print(f"Dataset cache files: {dataset_cache_files[0]['filename']}")  # Print the path to the first cache file

# Optionally, print some sample data
print("Sample data:", dataset['train'][0])
