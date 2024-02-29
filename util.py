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