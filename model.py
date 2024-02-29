from util import get_data, encode_text, decode_text
import torch

encoded_text, encoder, decoder = get_data()

data = torch.tensor(encoded_text, dtype=torch.long)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 8
print(train_data[:block_size + 1])