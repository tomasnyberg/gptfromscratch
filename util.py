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

class Feedforward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class Head(nn.Module):
    # Head of self-attention

    def __init__(self, head_size, n_embed, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # attention scores, "affinities"
        wei = q @ k.transpose(-2, -1) * (1.0 / C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        att = wei @ v
        return att

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4, n_embed, block_size)
        self.ffwd = Feedforward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.device = torch.device('cpu')
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = token_embeddings + pos_emb
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)

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
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

