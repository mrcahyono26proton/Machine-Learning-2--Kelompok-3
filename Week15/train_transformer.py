import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import csv
import matplotlib.pyplot as plt

from util import load_vocab, collate_batch, evaluate_bleu, decode_ids, to_ids, PAD, BOS, EOS, UNK
from transformer import Transformer
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder

class NMTDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.data = [(to_ids(src, src_vocab), to_ids(trg, trg_vocab)) for src, trg in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)

def epoch_run(model, loader, criterion, optimizer, train=True):
    model.train() if train else model.eval()
    
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(train):
        for src, trg in tqdm(loader):
            src, trg = src.to(model.device), trg.to(model.device)

            output, _ = model(src, trg[:-1, :])
            logits = output.reshape(-1, output.size(-1))
            target = trg[1:].reshape(-1)
            
            loss = criterion(logits, target)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n_tokens = (target != PAD).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl

def plot_curves(history, save_prefix="transformer", fontsize=14):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val")
    plt.title("Cross-Entropy Loss per Epoch", fontsize=fontsize + 2)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=180)
    plt.show()

    # Perplexity
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_ppl"], marker="o", label="Train")
    plt.plot(epochs, history["val_ppl"], marker="o", label="Val")
    plt.title("Perplexity (PPL) per Epoch", fontsize=fontsize + 2)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.ylabel("PPL", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_ppl.png", dpi=180)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and vocabs
    with open("processed_data.json") as f:
        pairs_data = json.load(f)
    train_pairs, val_pairs, test_pairs = pairs_data["train"], pairs_data["val"], pairs_data["test"]
    
    en_vocab, en_itos = load_vocab("en_vocab.json")
    id_vocab, id_itos = load_vocab("id_vocab.json")

    # Create datasets and dataloaders
    train_ds = NMTDataset(train_pairs, en_vocab, id_vocab)
    val_ds   = NMTDataset(val_pairs, en_vocab, id_vocab)
    test_ds  = NMTDataset(test_pairs, en_vocab, id_vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    
    # Model hyperparameters
    SRC_VOCAB_SIZE = len(en_vocab)
    TRG_VOCAB_SIZE = len(id_vocab)
    EMBED_DIM = 256
    NUM_HEADS = 8
    FF_DIM = 512
    NUM_LAYERS = 3
    DROPOUT = 0.1

    # Instantiate the model
    encoder = TransformerEncoder(SRC_VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM, DROPOUT).to(device)
    decoder = TransformerDecoder(TRG_VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM, DROPOUT).to(device)
    model = Transformer(encoder, decoder, PAD, PAD, BOS, EOS, device).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": [], "val_bleu": []}
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = epoch_run(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_ppl = epoch_run(model, val_loader, criterion, optimizer, train=False)
        val_bleu = evaluate_bleu(model, val_loader, en_itos, id_itos, en_vocab)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)
        history["val_bleu"].append(val_bleu)

        print(f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} PPL {train_ppl:.2f} | Val Loss {val_loss:.4f} PPL {val_ppl:.2f} | Val Bleu {val_bleu:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "transformer_best.pt")
            print("Saving best model checkpoint.")

    # Save training history to CSV
    with open("transformer_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_ppl", "val_ppl", "val_bleu"])
        for i in range(args.epochs):
            w.writerow([i + 1, history["train_loss"][i], history["val_loss"][i], history["train_ppl"][i], history["val_ppl"][i], history["val_bleu"][i]])

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("transformer_best.pt", weights_only=True, map_location=device))
    test_loss, test_ppl = epoch_run(model, test_loader, criterion, optimizer, train=False)
    test_bleu = evaluate_bleu(model, test_loader, en_itos, id_itos, en_vocab)
    print(f"\nTEST  | Loss {test_loss:.4f} | PPL {test_ppl:.2f} | BLEU {test_bleu:.2f}")
    
    model.eval()
    n_show = 5
    shown = 0
    with torch.no_grad():
        for src, trg in test_loader:
            src = src.to(device)
            trg = trg.to(device)
            ys, _atts = model.greedy_decode(src, max_len=40)
            B = src.size(1)
            for b in range(min(B, n_show - shown)):
                src_txt = decode_ids(src[:, b], en_itos)
                trg_txt = decode_ids(trg[:, b], id_itos)
                pred_txt = decode_ids(ys[:, b], id_itos, src[:, b], en_itos)
                print("-" * 60)
                print("SRC :", src_txt)
                print("TRG :", trg_txt)
                print("PRED:", pred_txt)
                shown += 1
            if shown >= n_show:
                break

    plot_curves(history, save_prefix="transformer")

    print("Contoh prediksi:")
    for b in range(min(5, src.size(1))):
        src_txt = decode_ids(src[:, b], en_itos)
        trg_txt = decode_ids(trg[:, b], id_itos)
        pred_txt = decode_ids(ys[:, b], id_itos, src[:, b], en_itos)
        print(f"SRC: {src_txt}")
        print(f"TRG: {trg_txt}")
        print(f"PRED: {pred_txt}")
        print("-" * 20)
    
if __name__ == "__main__":
    main()