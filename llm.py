# Import necessary libraries
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.nn").setLevel(logging.ERROR)
logging.getLogger("torch.nn.functional").setLevel(logging.ERROR)

# Initialize a blank spacy model for tokenization
nlp = spacy.blank('en')

# Define a sample training dataset and a prompt for prediction
training_dataset = "Dr.Einstine discovered theory of relativity in 1,906 AD."
prompt = "Dr.Einstine discovered theory"

# Tokenize the prompt text using spacy
doc_of_prompt = nlp(prompt)
tokens_of_prompt = []

# Store tokens from the prompt text into a list
for token in doc_of_prompt:
    tokens_of_prompt.append(str(token))

# Tokenize the training dataset text using spacy
doc_of_training_dataset = nlp(training_dataset)
tokens_of_training_dataset = []

# Store tokens from the training dataset into a list
for token in doc_of_training_dataset:
    tokens_of_training_dataset.append(str(token))

# Create dictionaries (stoi: string-to-index) for the prompt and training dataset tokens
stoi_of_prompt = {token:i for i, token in enumerate(set(tokens_of_prompt))}
token_ids_of_prompt = [stoi_of_prompt[token] for token in tokens_of_prompt]

stoi_of_training_dataset = {token:i for i, token in enumerate(set(tokens_of_training_dataset))}
token_ids_of_training_dataset = [stoi_of_training_dataset[token] for token in tokens_of_training_dataset]

# Convert token IDs into tensors (PyTorch tensors for neural network processing)
tensor_of_training_dataset = torch.tensor(token_ids_of_training_dataset)
tensor_of_prompt = torch.tensor(token_ids_of_prompt)

# Determine the maximum length between the prompt and training dataset for padding
max_len = max(tensor_of_prompt.size(0), tensor_of_training_dataset.size(0))

# Pad both tensors to make them of equal length (to prevent dimension mismatch)
tensor_of_training_dataset = F.pad(tensor_of_training_dataset, (0, max_len - tensor_of_training_dataset.size(0)))
tensor_of_prompt = F.pad(tensor_of_prompt, (0, max_len - tensor_of_prompt.size(0)))

# Initialize an embedding layer for the training dataset tokens
embedding_layer = nn.Embedding(num_embeddings = len(tokens_of_training_dataset), embedding_dim = 16)
word_embedding = embedding_layer(tensor_of_training_dataset)

# Create a MultiheadAttention layer to compute attention weights for the training dataset
attn = nn.MultiheadAttention(embed_dim = 16, num_heads = 2, batch_first = False)
attention_of_training_dataset, weights = attn(word_embedding, word_embedding, word_embedding)

# Initialize an embedding layer for the prompt tokens
embedding_layer = nn_
