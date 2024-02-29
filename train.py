import torch
from model import encoded_text, get_batch, vocab_size, n_embed, block_size, lr, load_model, save_model, estimate_loss

max_iters = 100
eval_interval = 1000
eval_iters = 200

if __name__ == '__main__':
    data = torch.tensor(encoded_text, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    xb, yb = get_batch(train_data)

    epoch_to_load = ""  # Choose what epoch to actually load
    model, optimizer, epoch = load_model(None, None, epoch_to_load)
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
