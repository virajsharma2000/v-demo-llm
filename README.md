# Token Prediction with Multihead Attention

This project uses a neural network-based approach to predict the next token in a given text prompt. The model performs tokenization, embedding, attention mechanism, and prediction using a simple setup built with PyTorch and SpaCy.

## Overview

The goal of this project is to predict the next token in a given text using the **Multihead Attention** mechanism. The following steps are performed:

1. **Text Tokenization**: The input text (training dataset and prompt) is tokenized using SpaCy.
2. **Token-to-ID Mapping**: A mapping from tokens to integer IDs is created.
3. **Word Embedding**: The token IDs are converted into word embeddings using a PyTorch embedding layer.
4. **Multihead Attention**: The attention mechanism is applied to both the prompt and training dataset to capture the relationships between the tokens.
5. **Probability Calculation**: The attention outputs are passed through softmax to convert them into probabilities.
6. **Prediction**: The token with the highest probability is selected as the predicted next token.

## Requirements

To run this project, you will need the following Python packages:

* `spacy`: For tokenization of input text.
* `torch`: For creating and training the neural network model.
* `logging`: For handling logging and suppressing warnings.

You can install the required dependencies using `pip`:

```bash
pip install spacy torch
```

Additionally, you will need to download the SpaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Code Walkthrough

### Step 1: Tokenization

The input text (prompt and training dataset) is processed using **SpaCy**, a natural language processing library. SpaCy splits the text into tokens (words or punctuation), which are then stored as strings.

```python
nlp = spacy.blank('en')

# Tokenize the prompt
doc_of_prompt = nlp(prompt)
tokens_of_prompt = [str(token) for token in doc_of_prompt]

# Tokenize the training dataset
doc_of_training_dataset = nlp(training_dataset)
tokens_of_training_dataset = [str(token) for token in doc_of_training_dataset]
```

### Step 2: Token-to-ID Mapping

For each unique token, an integer ID is assigned using a dictionary. This step is necessary to convert the textual tokens into numerical representations that can be processed by the neural network.

```python
stoi_of_prompt = {token: i for i, token in enumerate(set(tokens_of_prompt))}
token_ids_of_prompt = [stoi_of_prompt[token] for token in tokens_of_prompt]

stoi_of_training_dataset = {token: i for i, token in enumerate(set(tokens_of_training_dataset))}
token_ids_of_training_dataset = [stoi_of_training_dataset[token] for token in tokens_of_training_dataset]
```

### Step 3: Word Embedding

We use an **embedding layer** to convert the token IDs into dense vectors (word embeddings). These embeddings help represent words in a continuous vector space.

```python
embedding_layer = nn.Embedding(num_embeddings=len(tokens_of_training_dataset), embedding_dim=16)
word_embedding = embedding_layer(tensor_of_training_dataset)
```

### Step 4: Multihead Attention

The **Multihead Attention** mechanism is used to compute the relationships between tokens in the input sequences. We use two separate attention layers: one for the prompt and one for the training dataset.

```python
attn = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=False)
attention_of_training_dataset, weights = attn(word_embedding, word_embedding, word_embedding)

attn = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=False)
attention_of_prompt, weights = attn(word_embedding, word_embedding, word_embedding)
```

### Step 5: Softmax and Probability Calculation

The output of the attention layers is passed through **softmax** to convert the attention scores into probabilities. These probabilities indicate the likelihood of each token in the sequence.

```python
probs_of_training_dataset = F.softmax(attention_of_training_dataset, dim=-1)
probs_of_prompt = F.softmax(attention_of_prompt, dim=-1)
```

### Step 6: Prediction

The probabilities from both the prompt and training dataset are concatenated. The token with the highest probability is selected as the predicted token.

```python
combined_probs = torch.cat((probs_of_prompt, probs_of_training_dataset), dim=-1)
predicted_token_id = torch.argmax(combined_probs, dim=-1)
```

Finally, the predicted token is mapped back to its original string representation using a reverse dictionary (`reverse_stoi`), and it is printed out.

```python
reverse_stoi = {v: k for k, v in stoi_of_training_dataset.items()}
predicted_token = reverse_stoi.get(predicted_token_id[0].item(), "Unknown")
print(f"Predicted next token: {predicted_token}")
```

## Example Output

If you run the code with the example dataset and prompt provided:

```plaintext
Predicted next token: of
```

This means the model predicts the next token after the prompt "Dr.Einstine discovered theory" to be "of".

## How It Works

* **Multihead Attention**: This mechanism allows the model to focus on different parts of the input simultaneously, which helps it capture long-range dependencies between tokens in the text.
* **Word Embeddings**: These dense vector representations allow the model to better understand the semantic relationships between different tokens.
* **Softmax**: This activation function converts the raw attention scores into probabilities, helping the model select the most likely next token.
