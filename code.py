from typing import Union, Iterable, Callable
import random

import torch
import torch.nn as nn
import random
import math

def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]

def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]

### 1.1 Batching, shuffling, iteration
def build_loader(
    data_dict: dict, batch_size: int = 64, shuffle: bool = False
) -> Callable[[], Iterable[dict]]:
    
    first_key = next(iter(data_dict.keys()))
    data_length = len(data_dict[first_key])
    
    def loader():   
        indices = list(range(data_length))
        if shuffle:
            random.shuffle(indices)
        
        for i in range(math.ceil(data_length / batch_size)):
            start = i * batch_size
            end = min(start + batch_size, data_length)
            batch_indices = indices[start:end]
            batch = {key: [value[j] for j in batch_indices] for key, value in data_dict.items()}
            yield batch
            
    return loader

### 1.2 Converting a batch into inputs
def convert_to_tensors(text_indices: "list[list[int]]") -> torch.Tensor:
    
    longest_sentence = max([len(tokenized_sentence) for tokenized_sentence in text_indices])
    result = []
    
    for tokenized_sentence in text_indices:
        tokenized_sentence = tokenized_sentence + [0]*(longest_sentence - len(tokenized_sentence))
        result.append(tokenized_sentence)
    
    return torch.tensor(result, dtype=torch.int32)

### 2.1 Design a logistic model with embedding and pooling
def max_pool(x: torch.Tensor) -> torch.Tensor:

    return torch.max(x, dim = 1).values

class PooledLogisticRegression(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()

        self.embedding = embedding
        self.sigmoid = nn.Sigmoid()
        self.layer_pred = nn.Linear(self.embedding.embedding_dim * 2, 1)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()

        embedded_premise = emb(premise)
        embedded_hypothesis = emb(hypothesis)
                
        pooled_premise = max_pool(embedded_premise)
        pooled_hypothesis = max_pool(embedded_hypothesis)
        
        concatenated_embeddings = torch.cat((pooled_premise, pooled_hypothesis), 1)
        predictions = layer_pred(concatenated_embeddings)
        sigmoid_predictions = sigmoid(predictions)
        probabilities = torch.squeeze(sigmoid_predictions, 1)
        
        return probabilities

### 2.2 Choose an optimizer and a loss function
def assign_optimizer(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    return optimizer

def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    losses = -1 * (y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
    
    bce_loss = losses.mean()
    
    return bce_loss

### 2.3 Forward and backward pass
def forward_pass(model: nn.Module, batch: dict, device="cpu"):
    
    tensor_premise, tensor_hypothesis = convert_to_tensors(batch["premise"]), convert_to_tensors(batch["hypothesis"])
    
    tensor_premise = tensor_premise.to(device)
    tensor_hypothesis = tensor_hypothesis.to(device)
    model.to(device)
    
    return model(tensor_premise, tensor_hypothesis)

def backward_pass(
    optimizer: torch.optim.Optimizer, y: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
        
    optimizer.zero_grad()
    loss = bce_loss(y, y_pred)
    loss.backward()
    optimizer.step()
    
    return loss

### 2.4 Evaluation
def f1_score(y: torch.Tensor, y_pred: torch.Tensor, threshold=0.5) -> torch.Tensor:
    
    if threshold:
        y_pred = torch.where(y_pred > 0.5, torch.tensor(1), torch.tensor(0))
    
    #To avoid division by zero
    eps = 1e-8
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y)):
        real_label = y[i]
        predicted_label = y_pred[i]
        
        if real_label and predicted_label: 
            TP += 1
        elif real_label and not predicted_label:
            FN += 1
        elif not real_label and predicted_label:
            FP += 1
        elif not real_label and not predicted_label:
            TN += 1
    
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    
    f1_score= 2 / ((1/precision) + (1/recall))
    
    return torch.tensor(f1_score)

### 2.5 Train loop
def eval_run(
    model: nn.Module, loader: Callable[[], Iterable[dict]], device: str = "cpu"
) -> torch.Tensor:
    
    model.to(device)
    
    y_true = torch.tensor([])
    y_pred = torch.tensor([])
    
    for batch in loader():
        true_labels = batch["label"]
        y_true = torch.cat((y_true, torch.tensor(true_labels)))
        
        predictions = forward_pass(model, batch)
        y_pred = torch.cat((y_pred, predictions))
    
    return torch.tensor(y_true), torch.tensor(y_pred)

def train_loop(
    model: nn.Module,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs: int = 3,
    device: str = "cpu",
):

    model.to(device)
    
    f1_scores = []
    losses = []
    
    for _ in range(n_epochs): 
        
        model.train()
        
        for batch in train_loader():
            y_true = torch.tensor(batch["label"])
            predictions = forward_pass(model, batch)
            loss = backward_pass(optimizer, y_true, predictions)
            losses.append(loss.item())
        
        model.eval()

        valid_true, valid_predictions = eval_run(model, valid_loader)
        score = f1_score(valid_true, valid_predictions)
        f1_scores.append(score)
            
        print(f"F1 score: {score}")
        
    return f1_scores

### 3.1
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int):
        super().__init__()

        self.embedding = embedding
        self.ff_layer = nn.Linear(self.embedding.embedding_dim * 2, hidden_size)
        self.layer_pred = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layer(self):
        return self.ff_layer

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layer = self.get_ff_layer()
        act = self.get_activation()

        embedded_premise = emb(premise)
        embedded_hypothesis = emb(hypothesis)
                
        pooled_premise = max_pool(embedded_premise)
        pooled_hypothesis = max_pool(embedded_hypothesis)
        
        concatenated_embeddings = torch.cat((pooled_premise, pooled_hypothesis), 1)
        hidden_layer = ff_layer(concatenated_embeddings)
        activated_hidden_layer = act(hidden_layer)
        predictions = layer_pred(activated_hidden_layer)
        sigmoid_predictions = sigmoid(predictions)
        probabilities = torch.squeeze(sigmoid_predictions, 1)
        
        return probabilities

### 3.2
class DeepNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, num_layers: int = 2):
        super().__init__()

        self.embedding = embedding
        self.ff_layers = nn.ModuleList()
        self.ff_layers.append(nn.Linear(self.embedding.embedding_dim * 2, hidden_size))
        for _ in range(num_layers - 1):
            self.ff_layers.append(nn.Linear(hidden_size, hidden_size))
            
        self.layer_pred = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()
        
    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layers(self):
        return self.ff_layers

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layers = self.get_ff_layers()
        act = self.get_activation()

        embedded_premise = emb(premise)
        embedded_hypothesis = emb(hypothesis)
                
        pooled_premise = max_pool(embedded_premise)
        pooled_hypothesis = max_pool(embedded_hypothesis)
        
        concatenated_embeddings = torch.cat((pooled_premise, pooled_hypothesis), 1)
        hidden_layer = concatenated_embeddings
        
        for h_layer in ff_layers:
            hidden_layer = act(h_layer(hidden_layer))
            
        predictions = layer_pred(hidden_layer)
        sigmoid_predictions = sigmoid(predictions)
        probabilities = torch.squeeze(sigmoid_predictions, 1)
        
        return probabilities

if __name__ == "__main__":
    # If you have any code to test or train your model, do it BELOW!

    # Seeds to ensure reproducibility
    random.seed(2022)
    torch.manual_seed(2022)

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data")

    train_tokens = {
        "premise": tokenize(train_raw["premise"], max_length=64),
        "hypothesis": tokenize(train_raw["hypothesis"], max_length=64),
    }

    valid_tokens = {
        "premise": tokenize(valid_raw["premise"], max_length=64),
        "hypothesis": tokenize(valid_raw["hypothesis"], max_length=64),
    }

    word_counts = build_word_counts(
        train_tokens["premise"]
        + train_tokens["hypothesis"]
        + valid_tokens["premise"]
        + valid_tokens["hypothesis"]
    )
    
    index_map = build_index_map(word_counts, max_words=10000)

    train_indices = {
        "label": train_raw["label"],
        "premise": tokens_to_ix(train_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(train_tokens["hypothesis"], index_map)
    }

    valid_indices = {
        "label": valid_raw["label"],
        "premise": tokens_to_ix(valid_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(valid_tokens["hypothesis"], index_map)
    }
    
    # 1.1
    train_loader1 = build_loader(train_indices, 128, True)
    valid_loader1 = build_loader(valid_indices, 128, True)
    
    # 1.2
    #batch = next(train_loader())
    
    # 2.1
    embedding1 = nn.Embedding(10000, 32, padding_idx=0)
    model1 = PooledLogisticRegression(embedding1) 
    optimizer1 = assign_optimizer(model1, lr=0.001)
    f1_scores1 = train_loop(model1, train_loader1, valid_loader1, optimizer1, n_epochs=8, device=device)
    
    train_loader2 = build_loader(train_indices, 128, True)
    valid_loader2 = build_loader(valid_indices, 128, True)
    
    embedding2 = nn.Embedding(10000, 32, padding_idx=0)
    model2 = ShallowNeuralNetwork(embedding2, hidden_size=64)
    optimizer2 = assign_optimizer(model2, lr=0.0005)
    f1_scores2 = train_loop(model2, train_loader2, valid_loader2, optimizer2, n_epochs=8, device=device)
    
    train_loader3 = build_loader(train_indices, 128, True)
    valid_loader3 = build_loader(valid_indices, 128, True)
    
    embedding3 = nn.Embedding(10000, 32, padding_idx=0)
    model3 = DeepNeuralNetwork(embedding3, hidden_size=64, num_layers=3)
    optimizer3 = assign_optimizer(model3, lr=0.0005)
    f1_scores3 = train_loop(model3, train_loader3, valid_loader3, optimizer3, n_epochs=8, device=device)

    # 2.2
    optimizer = "your code here"

    # 2.4
    score = "your code here"

    # 2.5
    n_epochs = 2

    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.1
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.2
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"