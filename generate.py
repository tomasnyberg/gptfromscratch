from model import vocab_size, n_embed, block_size, lr, decode, load_model
import torch

epoch_to_load = "1000"

model, optimizer, _ = load_model(None, None, epoch_to_load)
print(f"Loaded model from checkpoint {epoch_to_load}!")
print("Generating shakespeare impression...")
print("--------------\n\n")
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))