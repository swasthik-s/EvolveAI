"""UTF-8 byte tokenizer and next-token dataset utilities. No external tokenizer needed."""
import torch

PAD_ID, BOS_ID, EOS_ID = 256, 257, 258
VOCAB_SIZE = 259

def encode(text, add_bos=False, add_eos=False):
    tokens = list(str(text).encode("utf-8"))
    if add_bos: tokens.insert(0, BOS_ID)
    if add_eos: tokens.append(EOS_ID)
    return tokens

def decode(tokens):
    raw = bytes(int(t) for t in tokens if 0 <= int(t) <= 255)
    return raw.decode("utf-8", errors="replace")

def prepare_text(text):
    # Separate documents with an end marker so examples don't silently merge.
    docs = [part.strip() for part in str(text).split("\n\n") if part.strip()]
    ids = []
    for doc in docs:
        ids.extend(encode(doc, add_bos=True, add_eos=True))
    return torch.tensor(ids, dtype=torch.long)

def split_data(tokens, train_fraction=0.9):
    if len(tokens) < 100:
        raise ValueError("Please provide at least 100 UTF-8 bytes of training text.")
    cut = max(1, min(len(tokens)-1, int(len(tokens) * train_fraction)))
    train, val = tokens[:cut], tokens[cut:]
    if len(val) < 2: val = train[-min(len(train), 16):]
    return train, val

def get_batch(data, batch_size, block_size, device="cpu"):
    if len(data) <= block_size:
        # Repeat tiny corpora to allow a batch while clearly warning in the UI.
        data = data.repeat((block_size * 2 // len(data)) + 2)
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in starts])
    y = torch.stack([data[i+1:i+block_size+1] for i in starts])
    return x.to(device), y.to(device)
