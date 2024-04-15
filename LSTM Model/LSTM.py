# -*- coding: utf-8 -*-
"""
# 1. Libraries
"""

!pip install torchdata
!pip install portalocker
import torch
import torch.nn.functional as F
import torchtext

train_iter, test_iter = torchtext.datasets.IMDB(split = ('train', 'test'))
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

""" LSTM Model """

MODEL_NAME = "imdb-rnn.model"
EPOCH = 10
BATCHSIZE = 64
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
# 2. Load Data
"""

train_data = [(label, tokenizer(line)) for label, line in train_iter]
train_data.sort(key = lambda x: len(x[1]))

test_data = [(label, tokenizer(line)) for label, line in test_iter]
test_data.sort(key = lambda x: len(x[1]))

for i in range(10):
  print(train_data[i])

def make_vocab(train_data, min_freq):
  vocab = {}
  for label, tokenlist in train_data:
    for token in tokenlist:
      if token not in vocab:
        vocab[token] = 0
      vocab[token] += 1

  vocablist = [('<unk>', 0), ('<pad', 0), ('<cls>', 0), ('<eos>', 0)]
  vocabidx = {}

  for token, freq in vocab.items():
    if freq >= min_freq:
      idx = len(vocablist)
      vocablist.append((token, freq))
      vocabidx[token] = idx

  vocabidx['<unk>'] = 0
  vocabidx['<pad>'] = 1
  vocabidx['<cls>'] = 2
  vocabidx['<eos>'] = 3
  return vocablist, vocabidx

vocablist, vocabidx = make_vocab(train_data, 10)

def preprocess(data, vocabidx):
  rr = []
  for label, tokenlist in data:
    tkl = ['<cls>']
    for token in tokenlist:
      tkl.append(token if token in vocabidx else '<unk>')
    tkl.append('<eos>')
    rr.append((label, tkl))
  return rr

train_data = preprocess(train_data, vocabidx)
test_data = preprocess(test_data, vocabidx)

for i in range(10):
  print(train_data[i])

def make_batch(data, batchsize):
  bb = []
  blabel = []
  btokenlist = []
  for label, tokenlist in data:
    blabel.append(label)
    btokenlist.append(tokenlist)
    if len(blabel) >= batchsize:
      bb.append((btokenlist, blabel))
      blabel = []
      btokenlist = []
  if len(blabel) > 0:
    bb.append((btokenlist, blabel))
  return bb

train_data = make_batch(train_data, BATCHSIZE)
test_data = make_batch(test_data, BATCHSIZE)

def padding(bb):
  for tokenlists, labels in bb:
    maxlen = max([len(x) for x in tokenlists])
    for tkl in tokenlists:
      for i in range(maxlen - len(tkl)):
        tkl.append('<pad>')
  return bb

train_data = padding(train_data)
test_data = padding(test_data)

def word2id(bb, vocabidx):
  rr = []
  for tokenlists, labels in bb:
    id_labels = [label -1 for label in labels]
    id_tokenlists = []
    for tokenlist in tokenlists:
      id_tokenlists.append([vocabidx[token] for token in tokenlist])
    rr.append((id_tokenlists, id_labels))
  return rr

train_data = word2id(train_data, vocabidx)
test_data = word2id(test_data, vocabidx)

class MyLSTM(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(MyLSTM, self).__init__()
    self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
    self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
    self.fc = torch.nn.Linear(hidden_dim, output_dim)

  def forward(self, text):
    embedded = self.embedding(text)
    output, (hidden, cell) = self.lstm(embedded)
    return self.fc(hidden[-1])
    
"""# Train"""

def train():
  model = MyLSTM(len(vocablist), embedding_dim=300, hidden_dim=256, output_dim=2).to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  for epoch in range(EPOCH):
    loss = 0
    for tokenlists, labels in train_data:
      optimizer.zero_grad()
      tokenlists = torch.tensor(tokenlists, dtype=torch.long).transpose(0, 1).to(DEVICE)
      labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)
      y = model(tokenlists)
      batchloss = F.cross_entropy(y.squeeze(), labels)
      batchloss.backward()
      optimizer.step()
      loss = loss + batchloss.item()
    print("epoch", epoch, ": loss", loss)
  torch.save(model.state_dict(), MODEL_NAME)

train()

"""# Test"""

def test():
  total = 0
  correct = 0
  model = MyLSTM(len(vocablist), embedding_dim=300, hidden_dim=256, output_dim=2).to(DEVICE)
  model.load_state_dict(torch.load(MODEL_NAME))
  model.eval()

  with torch.no_grad():
    for tokenlists, labels in test_data:
      tokenlists = torch.tensor(tokenlists, dtype=torch.long).transpose(0, 1).to(DEVICE)
      labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)
      y = model(tokenlists)
      pred_labels = torch.argmax(y, dim=1)
      correct += (pred_labels == labels).sum().item()
      total += len(labels)

  print("correct:", correct)
  print("total:", total)
  print("accuracy:", correct / float(total))

test()
