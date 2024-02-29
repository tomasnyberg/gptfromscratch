from util import get_data, get_batch, BigramLanguageModel, encode_text_with_encoder, decode_text_with_decoder
import torch

encoded_text, encoder, decoder = get_data()
vocab_size = len(encoder)

def encode(text):
    return encode_text_with_encoder(text, encoder)

def decode(encoded_text): 
    return decode_text_with_decoder(encoded_text, decoder)

data = torch.tensor(encoded_text, dtype=torch.long)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

batch_size = 4
block_size = 8

xb, yb = get_batch(train_data, batch_size, block_size)
model = BigramLanguageModel(vocab_size)

logits, loss = model(xb, yb)
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))