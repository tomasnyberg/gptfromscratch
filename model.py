def get_text():
    with open('input.txt', 'r') as f:
        text = f.read()
    return text

def get_vocab(text):
    return sorted(list(set(text)))

text = get_text()
vocab = get_vocab(text)
print(''.join(vocab))