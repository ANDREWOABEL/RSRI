"""
LSTM model for text classification

Input: Text
Output: Score between 0 and 1 of fake/real
"""
# ------------------------------------------------------------------
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
emb_dim = 128
hidden_dim = 128
learning_rate = 0.001
batch_size = 64
num_epochs = 2
num_layers = 1
num_classes = 2

class NewsDataset(Dataset):

	def __init__(self):
		
		with open('news-processed.csv', newline='') as f:
			reader = csv.reader(f)
			data = list(reader)
			del data[0]

		labels, text = zip(*data)

		# Determining vocabulary
		from itertools import chain
		vocab = sorted(set(chain.from_iterable([h.split() for h in text])))
		pad_id = len(vocab)
		bos_id = len(vocab) + 1
		eos_id = len(vocab) + 2
		vocab_size = eos_id + 1

		# B x T
		tokenized_text = [
			[bos_id] + [vocab.index(word) for word in h.split()] + [eos_id] for h in text
		]

		max_len = max(len(h) for h in tokenized_text)
		sequence_length = max_len

		padded_input = [
			h + (max_len - len(h)) * [pad_id]
			for h in tokenized_text
		]

		lstm_input = torch.LongTensor(padded_input)
		input_embedding = nn.Embedding(vocab_size, emb_dim)
		# B x T x E
		emb_inputs = input_embedding(lstm_input)

		labels = [int(i) for i in labels]
		labels_tensor = torch.LongTensor(labels)

		# B x T
		self.labels_tensor = labels_tensor
		self.emb_inputs = emb_inputs
		self.n_samples = labels_tensor.shape[0]
		self.sequence_length = max_len

	def __getitem__(self, index):
		return self.labels_tensor[index], self.emb_inputs[index]

	def __len__(self):
		return self.n_samples

	def seq_length(self):
		return self.sequence_length

news_dataset = NewsDataset()
test_dataset, train_dataset = torch.utils.data.random_split(44898,[8980,35918],generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

# MODEL

class BiLSTM(nn.Module):
	def __init__(self, emb_dim, hidden_dim, num_layers, num_classes):
		super(BiLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.num_classes = num_classes
		self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=True)
		self.scoring_head = nn.Linear(2 * hidden_dim, 1)

	def forward(self,emb_inputs):
		h0 = torch.zeros(self.num_layers*2, emb_inputs.size(0), hidden_dim).to(device) 
		c0 = torch.zeros(self.num_layers*2, emb_inputs.size(0), hidden_dim).to(device) 
		# B x T x (Dir * H) -> B x T x 2H
		lstm_output, _ = self.lstm(emb_inputs,(h0,c0))
		# Take the average across your sequence
		# B x T x 2H -> B x 2H
		hidden_representation = lstm_output.mean(dim=1)
		# Compute your score
		score = self.scoring_head(hidden_representation)
		# Score between 0 and 1
		score_01 = torch.sigmoid(score)
		return score_01

model = BiLSTM(emb_dim,hidden_dim,num_layers,num_classes).to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

print(enumerate(train_loader))

# Training the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (labels,text) in enumerate(train_loader):
		print(text.shape)
		print(labels.shape)
		text = text.reshape(-1, train_dataset.seq_length(), emb_dim).to(device) 
		labels = torch.unsqueeze(labels,-1)
		labels = labels.type_as(text).to(device) 
		
		score = model(text)
		loss = F.binary_cross_entropy_with_logits(input=score, target=labels)

		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()

		if (i+1) % 1000 == 0:
			print(f'Epoch [{epoch+1}/{num_epochs}]')
			print(f'Step [{i+1}/{n_total_steps}]')
			print(f'Loss: {loss.item():.4f}')

		
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	for labels, text in test_loader:
		text = text.reshape(-1, train_dataset.seq_length(), emb_dim).to(device) 
		labels = torch.unsqueeze(labels,-1)
		labels = labels.type_as(text).to(device) 
		
		scores = model(text)
		# max returns (value ,index)
		_, predicted = torch.max(scores.data, 1)
		n_samples += labels.size(0)
		n_correct += (predicted == labels).sum().item()

	acc = 100.0 * n_correct / n_samples
	print(f'Accuracy of the network is: {acc} %')