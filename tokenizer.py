import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_text(filename):
    with open(filename, 'r') as f:
        return f.read()


def count_frequencies(s, counts):
    for i in range(len(s) - 1):
        pair = (s[i], s[i+1])
        counts[pair] = counts.get(pair, 0) + 1


def replace(tokens, merges):
    new_tokens = []
    i = 0
    curr = None
    while i < len(tokens):
        if curr == None:
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) in merges:
                curr = merges[(tokens[i], tokens[i+1])]
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        else:
            tup = (curr, tokens[i])
            if tup in merges:
                curr = merges[tup]
                i += 1
            else:
                new_tokens.append(curr)
                curr = None
    if curr != None:
        new_tokens.append(curr)
    return new_tokens


def compress(text, vocab_size=276):
    assert vocab_size > 256, "Vocab size must be greater than 256."
    num_merges = vocab_size - 256
    chunks = re.findall(GPT2_SPLIT_PATTERN, text)
    ids = [list(chunk.encode('utf8')) for chunk in chunks]
    per_time = 1
    merges = {}
    for i in range(num_merges):
        counts = {}
        for chunked_id in ids:
            count_frequencies(chunked_id, counts)
        pair = max(counts, key=counts.get)
        merges[pair] = 256 + i
        for i, chunked_id in enumerate(ids):
            ids[i] = replace(chunked_id, merges)
    ids = [item for sublist in ids for item in sublist]
    return ids, merges


class Tokenizer:

    def train(self, text, vocab_size):
        self.text = text
        self.vocab_size = vocab_size
        self.ids, self.merges = compress(
            text, vocab_size)
        self.vocab_size = vocab_size

    def load(self, filename):
        result = {}
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                fr, to = line.split(' -> ')
                to = to.replace('(', '').replace(')', '').split(',')
                if int(fr) >= 256:
                    result[(int(to[0]), int(to[1]))] = int(fr)
                line = f.readline()
        self.merges = result
        self.vocab_size = 256 + len(result)

    def encode(self, text):
        assert hasattr(
            self, 'merges'), "You must train the tokenizer before encoding."
        inted = list(map(int, text.encode('utf8')))
        return replace(inted, self.merges)

    def decode(self, tokens):
        assert hasattr(
            self, 'merges'), "You must train the tokenizer before encoding."
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return b''.join(vocab[token] for token in tokens).decode('utf8', errors='replace')

    def visualize_compression(self, filename=None, nice_print=True):
        assert hasattr(
            self, 'merges'), "You must train the tokenizer before visualizing."
        vocab = {idx: idx for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = (p0, p1)

        def find(x):
            if x <= 255:
                decoded = bytes([x]).decode('utf8', errors='replace')
                return (repr(decoded),) if not decoded.isprintable() else (decoded,)
            return find(vocab[x][0]) + find(vocab[x][1])
        if nice_print:
            printstr = '\n'.join([f"{k} -> {''.join(find(k))}" for k in vocab])
        else:
            printstr = '\n'.join([f"{k} -> {vocab[k]}" for k in vocab])
        if not filename:
            print(printstr)
        else:
            with open(filename, 'w') as f:
                f.write(printstr)


def test_replace():
    tokens = [1, 2, 3, 4, 5, 6]
    merges = {(1, 2): 7, (7, 3): 8}
    assert replace(tokens, merges) == [
        8, 4, 5, 6], f"Expected [8,4,5,6], got {replace(tokens, merges)}"


def test_encode_decode():
    # Test that we are able to something with merges over merged tokens and then decode it back.
    # In this case we compress the whole string "ABCDEF" to just the token [260] and then test
    # that we get it back properly
    text = "ABCDEF"
    merges = {(65, 66): 256, (256, 67): 257, (257, 68)
               : 258, (258, 69): 259, (259, 70): 260}
    bt = Tokenizer()
    bt.merges = merges
    encoded = bt.encode(text)
    assert (len(encoded) == 1)
    decoded = bt.decode(encoded)
    assert decoded == text, f"Expected {text}, got {decoded}"


test_encode_decode()
test_replace()

bt = Tokenizer()
# bt.load("txtfiles/1500shakespearetokens.txt")
text = get_text("txtfiles/shortinput.txt")
bt.train(text, 500)
encoded = bt.encode(text)
print(
    f"Text length: {len(text)}, Tokenized text length: {len(encoded)}, Compression ratio: {len(encoded)/len(text)}")
print(len(text))
print(len(encoded))
decoded = bt.decode(encoded)
assert text == decoded
