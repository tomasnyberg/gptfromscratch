text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokens = list(map(int, text.encode('utf8')))

with open('blogpost.txt', 'r') as f:
    blogtext = f.read()
    blogtokens = list(map(int, blogtext.encode('utf8')))


def most_frequent_pair(s):
    counts = {}
    for i in range(len(s) - 1):
        pair = (s[i], s[i+1])
        counts[pair] = counts.get(pair, 0) + 1
    return max(counts, key=counts.get)


def replace(tokens, pair, replacement):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            new_tokens.append(replacement)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def compress(tokens, vocab_size=276):
    assert vocab_size > 256, "Vocab size must be greater than 256."
    num_merges = vocab_size - 256
    ids = list(tokens)

    merges = {}
    for i in range(num_merges):
        pair = most_frequent_pair(ids)
        merges[pair] = 256 + i
        ids = replace(ids, pair, merges[pair])

    return ids, merges


def decode(compressed_tokens, merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return b''.join(vocab[token] for token in compressed_tokens).decode('utf8', errors='replace')


def encode(text, merges):
    inted = list(map(int, text.encode('utf8')))
    for (p0, p1), idx in merges.items():
        inted = replace(inted, (p0, p1), idx)
    return inted


class BasicTokenizer:

    def train(self, text, vocab_size):
        self.text = text
        self.vocab_size = vocab_size
        self.ids, self.merges = compress(list(map(int, text.encode('utf8'))), vocab_size)

    def encode(self, text):
        assert hasattr(
            self, 'merges'), "You must train the tokenizer before encoding."
        return encode(text, self.merges)

    def decode(self, tokens):
        assert hasattr(
            self, 'merges'), "You must train the tokenizer before encoding."
        return decode(tokens, self.merges)

    def visualize_compression(self, filename = None):
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
        printstr = '\n'.join([f"{k} -> {''.join(find(k))}" for k in vocab])
        if not filename:
            print(printstr)
        else:
            with open(filename, 'w') as f:
                f.write(printstr)


bt = BasicTokenizer()
bt.train(blogtext, 500)
bt.visualize_compression("tokens.txt")
encoded = bt.encode(blogtext)
decoded = bt.decode(encoded)
assert blogtext == decoded
