from util import get_data, get_batch, BigramLanguageModel
import torch

encoded_text, encoder, decoder = get_data()
vocab_size = len(encoder)

data = torch.tensor(encoded_text, dtype=torch.long)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

batch_size = 4
block_size = 8

xb, yb = get_batch(train_data, batch_size, block_size)
model = BigramLanguageModel(vocab_size)
out = model(xb, yb)
print(out)