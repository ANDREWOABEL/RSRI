"""
LSTM model for text classification

Input: Text
Output: Score between 0 and 1 of fake/real
"""
# ------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random
import nltk
from tqdm import tqdm
from itertools import chain
from nltk import word_tokenize
import pandas as pd
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NewsDataset(Dataset):
    def __init__(self, labels, text, bos_id, eos_id):
        self.labels = labels
        self.text = text
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ix):
        label = self.labels[ix]
        # Add BOS and EOS to text token sequence
        tokens = [self.bos_id] + self.text[ix] + [self.eos_id]
        return label, tokens

class NewsCollator():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        # labels[i] = label for observation i in the batch
        # tokens[i] = list of token IDs for observation i in the batch
        labels, tokens = zip(*batch)
        # Assess necessary padding for tokens
        max_seq_len = max(len(t) for t in tokens)
        padded_tokens = [
            t + [self.pad_id] * (max_seq_len - len(t)) for t in tokens
        ]
        # Padding mask: 1 for
        token_padding_masks = [[1] * len(t) + [0] * (max_seq_len - len(t))
                               for t in tokens]
        # Labels, input tokens, padding masks
        return torch.LongTensor(labels), torch.LongTensor(
            padded_tokens), torch.BoolTensor(token_padding_masks)

# MODEL
class BiLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers, num_classes,
                 vocab_size):
        super(BiLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        # Input embedding
        self.input_embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True)

        # Bidirectional
        self.scoring_head = nn.Linear(2 * self.hidden_dim, self.num_classes)

    def forward(self, input_tokens):
        # Embed imput tokens (B x T -> B x T x E)
        emb_inputs = self.input_embedding(input_tokens)
        # B x T x E -> B x T x (Dir * H = 2H)
        # H_0 and C_0 are implicitly set to 0 already
        lstm_output, _ = self.lstm(emb_inputs)
        # Take the average across your sequence
        # B x T x 2H -> B x 2H
        hidden_representation = lstm_output.mean(dim=1)
        # Compute your score logits: B x 2H -> B x Nc
        score = self.scoring_head(hidden_representation)
        return score

# Hyperparameters
emb_dim = 128
hidden_dim = 128
learning_rate = 0.001
batch_size = 64
num_epochs = 2
num_layers = 1
num_classes = 2

# Load raw data with 2 columns
file_loc = 'news-processed.csv'
raw_data = pd.read_csv(file_loc)

# Determining vocabulary - list of unique words
text_tokens = []
all_labels = []
vocab = set()

# Tokenize each sentence
for label, h in tqdm(
        zip(raw_data['label'].values, raw_data['text'].values),
        total=len(raw_data)):
    # Invalid values
    if not isinstance(h, str) or not h.strip():
        continue
    tokens = word_tokenize(h)
    vocab |= set(tokens)
    text_tokens.append(tokens)
    # Convert labels to boolean - 0 = real news, 1 = fake news
    all_labels.append(int(label == 2))

# Convert tokens into IDs
vocab_map = dict(zip(sorted(vocab), range(len(vocab))))
pad_id = len(vocab_map)
bos_id = len(vocab_map) + 1
eos_id = len(vocab_map) + 2
vocab_size = eos_id + 1
print(f'{len(vocab_map):,} tokens: vocabulary w {vocab_size:,} total tokens')
text_tokens = [[vocab_map[t] for t in tokens] for tokens in text_tokens]

# Split into training/test
indices = list(range(len(text_tokens)))
random.Random(42).shuffle(indices)

# 80% for training, 20% for evaluation
train_cutoff = int(len(indices) * 0.8)
train_indices = indices[:train_cutoff]
eval_indices = indices[train_cutoff:]

# Training data
train_ds = NewsDataset(
    labels=[all_labels[i] for i in train_indices],
    text=[text_tokens[i] for i in train_indices],
    bos_id=bos_id,
    eos_id=eos_id)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=NewsCollator(pad_id))

# Evaluation data
eval_ds = NewsDataset(
    labels=[all_labels[i] for i in eval_indices],
    text=[text_tokens[i] for i in eval_indices],
    bos_id=bos_id,
    eos_id=eos_id)
eval_loader = DataLoader(
    eval_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=NewsCollator(pad_id))

# Create model and load weights into optimizer
model = BiLSTM(emb_dim, hidden_dim, num_layers, num_classes,
               vocab_size).to(DEVICE)
print('{} parameters'.format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (labels, input_tokens, padding_mask) in tqdm(
            enumerate(train_loader), total=n_total_steps):
        # Move to correct device
        labels = labels.to(DEVICE)
        input_tokens = input_tokens.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)

        # Run model - score is B x Nc
        score = model(input_tokens)

        # Compute loss between B x Nc score and B labels
        loss = F.cross_entropy(input=score, target=labels)

        # Backwards step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Logging
        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Step [{i+1}/{n_total_steps}]')
            print(f'Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (labels, input_tokens, padding_mask) in tqdm(
            enumerate(eval_loader), total=len(eval_ds)):
        # Move to correct device
        labels = labels.to(DEVICE)
        input_tokens = input_tokens.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)

        # Run model
        score = model(input_tokens)
        predicted = torch.argmax(score, dim=-1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network is: {acc} %')
