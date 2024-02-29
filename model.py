from util import get_data, get_batch, BigramLanguageModel, encode_text_with_encoder, decode_text_with_decoder, estimate_loss
import torch

# Hyperparams
batch_size = 16
block_size = 64
max_iters = 5000
eval_interval = 100
lr = 1e-4
eval_iters = 200
n_embed = 16


encoded_text, encoder, decoder = get_data()
vocab_size = len(encoder)

def encode(text):
    return encode_text_with_encoder(text, encoder)

def decode(encoded_text): 
    return decode_text_with_decoder(encoded_text, decoder)

def checkpoint_path(iter):
    if iter == "":
        return 'bigram_language_model_checkpoint.pth'
    return f'bigram_language_model_checkpoint_{iter}.pth'

def save_model(model, optimizer, epoch, loss):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path(epoch))


if __name__ == '__main__':
    data = torch.tensor(encoded_text, dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    xb, yb = get_batch(train_data, batch_size, block_size)
    model = BigramLanguageModel(vocab_size, n_embed, block_size)

    # checkpoint_path = 'bigram_language_model_checkpoint.pth'

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epoch_to_load = "0"
    checkpoint = torch.load(checkpoint_path(epoch_to_load))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    for steps in range(max_iters):
        if steps % eval_interval == 0:
            losses = estimate_loss(model, block_size, batch_size, train_data, val_data, eval_iters)
            print(f"Step: {steps}, Train loss: {losses['train']}, Val loss: {losses['val']}")
            save_model(model, optimizer, steps + epoch, loss)
        xb, yb = get_batch(train_data, batch_size, block_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # checkpoint = {
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'epoch': epoch + steps,
    #     'loss': loss
    # }
    # torch.save(checkpoint, checkpoint_path)