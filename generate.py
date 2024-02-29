from util import BigramLanguageModel
from model import vocab_size, n_embed, block_size, lr, decode
import torch
model = BigramLanguageModel(vocab_size, n_embed, block_size)
checkpoint_path = 'bigram_language_model_checkpoint.pth'
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print("Loaded model from checkpoint")
print(f"Epoch: {epoch}, Loss: {loss}")

idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))