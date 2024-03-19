from model import n_embed, block_size, lr, load_model
from train import init_tokenizer
import torch

tokenizer = init_tokenizer(load=True, save=False)

epoch_to_load = "4000"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, optimizer, _ = load_model(None, None, epoch_to_load, tokenizer)
print(f"Loaded model from checkpoint {epoch_to_load}!")
print("Generating shakespeare impression...")
print("--------------\n\n")
idx = torch.zeros((1, 1), dtype=torch.long).to(device)
print(tokenizer.decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))
