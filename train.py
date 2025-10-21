# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from data_utils import (
    set_seed, list_matching_pairs, make_loaders,
    VOCAB_SIZE, BOS, EOS,
    read_file_bytes, encode_bytes_as_ids, add_bos_eos
)
from model import CharTransformer

# -------------------------
# Discriminator scaffold
# -------------------------
class Discriminator(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        emb = self.embed(x).transpose(1, 2)  # [B, H, T]
        feat = self.conv(emb).mean(dim=2)    # [B, C]
        return self.fc(feat)                 # [B, 1]

# -------------------------
# Save generated outputs
# -------------------------
def save_generated_outputs(model, input_dir, gen_dir, max_len, device):
    os.makedirs(gen_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for fname in os.listdir(input_dir):
            in_path = os.path.join(input_dir, fname)
            if not os.path.isfile(in_path):
                continue
            data = read_file_bytes(in_path, max_len)
            ids = add_bos_eos(encode_bytes_as_ids(data))
            src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            pred = model.greedy_decode(src, max_len=max_len)
            pred_bytes = bytes([b for b in pred[0].cpu().numpy() if b < 256])
            out_path = os.path.join(gen_dir, fname)
            with open(out_path, "wb") as f:
                f.write(pred_bytes)
    print(f"Generated outputs written to {gen_dir}")

# -------------------------
# Main training
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=Config.input_dir, help="Path to input folder")
    parser.add_argument("--output", default=Config.output_dir, help="Path to output folder")
    parser.add_argument("--gen", default="generated", help="Path to save generated outputs")
    parser.add_argument("--epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--batch", type=int, default=Config.batch_size)
    parser.add_argument("--maxlen", type=int, default=Config.max_seq_len)
    parser.add_argument("--save", default=Config.save_path)
    parser.add_argument("--mode", choices=["generator", "gan"], default="generator",
                        help="Training mode: 'generator' (only CE loss) or 'gan' (adversarial)")
    args = parser.parse_args()

    set_seed(Config.seed)

    pairs = list_matching_pairs(args.input, args.output)
    if len(pairs) < Config.min_pairs:
        raise RuntimeError(f"No matching pairs found in {args.input} and {args.output}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_loaders(pairs, args.maxlen, args.batch)

    # Generator
    generator = CharTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_encoder_layers=Config.num_encoder_layers,
        num_decoder_layers=Config.num_decoder_layers,
        dim_feedforward=Config.dim_feedforward,
        dropout=Config.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.mode == "generator":
        optimizer = optim.AdamW(generator.parameters(), lr=Config.learning_rate)

        for epoch in range(1, args.epochs + 1):
            generator.train()
            total_loss = 0.0
            for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}"):
                src, tgt = src.to(device), tgt.to(device)
                dec_in, dec_target = tgt[:, :-1], tgt[:, 1:]
                logits = generator(src, dec_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), Config.grad_clip)
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}: CE_loss={total_loss/len(train_loader):.4f}")

    elif args.mode == "gan":
        discriminator = Discriminator().to(device)
        g_optimizer = optim.AdamW(generator.parameters(), lr=Config.learning_rate)
        d_optimizer = optim.AdamW(discriminator.parameters(), lr=Config.learning_rate)
        bce = nn.BCELoss()

        for epoch in range(1, args.epochs + 1):
            generator.train()
            discriminator.train()
            total_g_loss, total_d_loss = 0.0, 0.0

            for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}"):
                src, tgt = src.to(device), tgt.to(device)

                # Generator step
                dec_in, dec_target = tgt[:, :-1], tgt[:, 1:]
                logits = generator(src, dec_in)
                ce_loss = criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))

                fake = generator.greedy_decode(src, max_len=args.maxlen)
                d_fake = discriminator(fake).squeeze()
                g_adv_loss = bce(d_fake, torch.ones_like(d_fake))

                g_loss = ce_loss + 0.1 * g_adv_loss
                g_optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                g_optimizer.step()

                # Discriminator step
                with torch.no_grad():
                    fake = generator.greedy_decode(src, max_len=args.maxlen)

                d_real = discriminator(tgt).squeeze()
                d_fake = discriminator(fake.detach()).squeeze()
                d_loss_real = bce(d_real, torch.ones_like(d_real))
                d_loss_fake = bce(d_fake, torch.zeros_like(d_fake))
                d_loss = (d_loss_real + d_loss_fake) / 2

                d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                d_optimizer.step()

                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()

            print(f"Epoch {epoch}: G_loss={total_g_loss/len(train_loader):.4f}, "
                  f"D_loss={total_d_loss/len(train_loader):.4f}")

    # Save generator checkpoint
    torch.save({"model": generator.state_dict()}, args.save)
    print(f"Saved generator checkpoint to {args.save}")

    # Generate outputs for all inputs
    save_generated_outputs(generator, args.input, args.gen, args.maxlen, device)

if __name__ == "__main__":
    main()