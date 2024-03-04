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


def compress(tokens):
    vocab_size = 276
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

encoded, merges = compress(tokens)

encoded = encode(blogtext, merges)
decoded = decode(encoded, merges)
print(blogtext == decoded)

# print(len(blogtext))
# print(len(encode(blogtext, merges)))
# print(len(decode(, merges)))