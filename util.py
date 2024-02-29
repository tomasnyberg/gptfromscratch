import torch

def get_text():
    with open('input.txt', 'r') as f:
        text = f.read()
    return text

def get_vocab(text):
    return sorted(list(set(text)))

# Character level tokenizer
def encoder_decoder(vocab):
    return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}

def encode_text(text, encoder):
    return [encoder[c] for c in text]

def decode_text(encoded_text, decoder):
    return ''.join([decoder[i] for i in encoded_text])

def get_data():
    text = get_text()
    vocab = get_vocab(text)
    encoder, decoder = encoder_decoder(vocab)
    encoded_text = encode_text(text, encoder)
    return encoded_text, encoder, decoder

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y