import torch
from model import get_batch, n_embed, block_size, lr, load_model, save_model, estimate_loss
from tokenizer import Tokenizer

INPUT_TEXT = "txtfiles/shortinput.txt"
TOKENIZER_INIT = "txtfiles/uglytokens.txt"
VOCAB_SIZE = 300
max_iters = 5000
eval_interval = 1000
eval_iters = 200



def get_input_text():
    with open(INPUT_TEXT, 'r') as file:
        return file.read()

def init_tokenizer(load=True, save=False):
    tokenizer = Tokenizer()
    if load:
        tokenizer.load(TOKENIZER_INIT)
    else:
        text = get_input_text()
        tokenizer.train(text, VOCAB_SIZE)
        if save:
            tokenizer.save(TOKENIZER_INIT)
    return tokenizer

if __name__ == '__main__':
    tokenizer = init_tokenizer(load=False, save=True)
    tokenized_text = tokenizer.encode(get_input_text())
    data = torch.tensor(tokenized_text, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    xb, yb = get_batch(train_data)

    epoch_to_load = ""  # Choose what epoch to actually load
    model, optimizer, epoch = load_model(None, None, epoch_to_load, tokenizer)
    for steps in range(max_iters):
        if steps % 10 == 0:
            print("Step:", steps)
        if steps % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, eval_iters)
            print(
                f"Step: {steps}, Train loss: {losses['train']}, Val loss: {losses['val']}")
            save_model(model, optimizer, steps + epoch, losses['val'])
        xb, yb = get_batch(train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    save_model(model, optimizer, steps + epoch, losses['val'])
