import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# Load data
data = pd.read_csv('IMDB Dataset.csv')
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train Word2Vec model
train_sentences = [review.split() for review in train_data['review']]
w2v_model = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=2, workers=4)


def vectorize_sentences(sentences, model, max_len):
    vec_sentences = []
    for sentence in sentences:
        vec_sentence = [model.wv[word] for word in sentence if word in model.wv]
        vec_sentence = vec_sentence[:max_len]
        if len(vec_sentence) < max_len:
            vec_sentence += [np.zeros(model.vector_size) for _ in range(max_len - len(vec_sentence))]
        vec_sentences.append(np.array(vec_sentence, dtype=np.float32))
    return np.array(vec_sentences, dtype=np.float32)


max_len = 100
X_train = vectorize_sentences(train_data['review'], w2v_model, max_len)
X_test = vectorize_sentences(test_data['review'], w2v_model, max_len)
Y_train = train_data['sentiment'].values
Y_test = test_data['sentiment'].values

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, hn, cn):
        # Create an input tensor filled with zeros for the decoder
        batch_size = hn.size(1)
        x = torch.zeros(batch_size, 1, self.hidden_dim).to(hn.device)
        out, (hn, cn) = self.lstm(x, (hn, cn))
        out = self.fc(out[:, -1, :])
        return out


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        hn, cn = self.encoder(x)
        output = self.decoder(hn, cn)
        return output


input_dim = 100
hidden_dim = 128
output_dim = 2
num_layers = 2

encoder = Encoder(input_dim, hidden_dim, num_layers)
decoder = Decoder(hidden_dim, output_dim, num_layers)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy_seq2seq = accuracy_score(all_labels, all_preds)
print(f'Seq2Seq Accuracy: {accuracy_seq2seq}')
print(f'Seq2Seq Classification Report:')
print(classification_report(all_labels, all_preds))
