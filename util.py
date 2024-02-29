import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(420)


def get_text():
    with open('input.txt', 'r') as f:
        text = f.read()
    return text


def get_vocab(text):
    return sorted(list(set(text)))

# Character level tokenizer


def encoder_decoder(vocab):
    return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}


def encode_text_with_encoder(text, encoder):
    return [encoder[c] for c in text]


def decode_text_with_decoder(encoded_text, decoder):
    return ''.join([decoder[i] for i in encoded_text])


def get_data():
    text = get_text()
    vocab = get_vocab(text)
    encoder, decoder = encoder_decoder(vocab)
    encoded_text = encode_text_with_encoder(text, encoder)
    return encoded_text, encoder, decoder


def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, block_size, batch_size, train_data, val_data, eval_iters):
    out = {}
    model.eval()
    for data, name in [(train_data, 'train'), (val_data, 'val')]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
