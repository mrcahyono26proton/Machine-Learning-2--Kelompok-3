import argparse
from pathlib import Path
import json

from util import load_pairs, split_pairs, to_ids, pad_batch
from analisis import build_vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/ind-eng/ind.txt', help='Path to txt data')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum sentence length')
    parser.add_argument('--max_vocab', type=int, default=None, help='Maximum vocabulary size')
    args = parser.parse_args()

    data_file = Path(args.data_path)

    # 1) Load, preprocess, and filter pairs
    pairs = load_pairs(data_file, max_len=args.max_len)
    print(f"Total usable pairs after filtering: {len(pairs):,}")

    # 2) Split 80/10/10
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.8, 0.1)
    print(f"Train: {len(train_pairs):,}, Val: {len(val_pairs):,}, Test: {len(test_pairs):,}")

    # 3) Build separate vocabs (English and Indonesian)
    en_vocab, en_itos = build_vocab([src for src, _ in train_pairs], max_size=args.max_vocab)
    id_vocab, id_itos = build_vocab([tgt for _, tgt in train_pairs], max_size=args.max_vocab)

    with open("en_vocab.json", "w") as f:
        json.dump(en_vocab, f)
    with open("id_vocab.json", "w") as f:
        json.dump(id_vocab, f)

    print(f"EN vocab size: {len(en_vocab):,} | ID vocab size: {len(id_vocab):,}")

    # Save pairs for later use in training script
    pairs_data = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }
    with open("processed_data.json", "w") as f:
        json.dump(pairs_data, f)

    print("Data preparation complete. Pairs and vocabularies saved.")

if __name__ == "__main__":
    main()