from util import get_data, get_batch, BigramLanguageModel, encode_text_with_encoder, decode_text_with_decoder, estimate_loss
import torch

# Hyperparams
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
lr = 0.001
eval_iters = 200
n_embed = 32


encoded_text, encoder, decoder = get_data()
vocab_size = len(encoder)

def encode(text):
    return encode_text_with_encoder(text, encoder)

def decode(encoded_text): 
    return decode_text_with_decoder(encoded_text, decoder)

data = torch.tensor(encoded_text, dtype=torch.long)

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

xb, yb = get_batch(train_data, batch_size, block_size)
model = BigramLanguageModel(vocab_size, n_embed, block_size)

logits, loss = model(xb, yb)
idx = torch.zeros((1,1), dtype=torch.long)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


for steps in range(max_iters):
    if steps % eval_interval == 0:
        losses = estimate_loss(model, block_size, batch_size, train_data, val_data, eval_iters)
        print(f"Step: {steps}, Train loss: {losses['train']}, Val loss: {losses['val']}")
    xb, yb = get_batch(train_data, batch_size, block_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))
print(loss.item())