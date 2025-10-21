# data_utils.py
import os
import torch
import random
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

BOS = 256
EOS = 257
VOCAB_SIZE = 258  # bytes 0..255 + BOS + EOS

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_matching_pairs(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    for fname in os.listdir(input_dir):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        if os.path.isfile(in_path) and os.path.isfile(out_path):
            pairs.append((in_path, out_path))
    return pairs

def read_file_bytes(path: str, max_len: int) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) > max_len - 2:  # leave room for BOS/EOS
        data = data[:max_len - 2]
    return data

def encode_bytes_as_ids(b: bytes) -> List[int]:
    # map raw bytes to 0..255
    return list(b)

def add_bos_eos(ids: List[int]) -> List[int]:
    return [BOS] + ids + [EOS]

class PairedTextDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], max_seq_len: int):
        self.samples = []
        for in_path, out_path in pairs:
            ib = read_file_bytes(in_path, max_seq_len)
            ob = read_file_bytes(out_path, max_seq_len)
            i_ids = add_bos_eos(encode_bytes_as_ids(ib))
            o_ids = add_bos_eos(encode_bytes_as_ids(ob))
            self.samples.append((torch.tensor(i_ids, dtype=torch.long),
                                 torch.tensor(o_ids, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_batch(batch):
    # pad to max lengths in batch
    in_seqs, out_seqs = zip(*batch)
    in_max = max(s.size(0) for s in in_seqs)
    out_max = max(s.size(0) for s in out_seqs)

    def pad_to(seqs, length):
        padded = torch.full((len(seqs), length), EOS, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :s.size(0)] = s
        return padded

    return pad_to(in_seqs, in_max), pad_to(out_seqs, out_max)

def make_loaders(pairs: List[Tuple[str, str]], max_seq_len: int, batch_size: int):
    # simple split
    random.shuffle(pairs)
    split = max(1, int(0.8 * len(pairs)))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_ds = PairedTextDataset(train_pairs, max_seq_len)
    val_ds = PairedTextDataset(val_pairs, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader