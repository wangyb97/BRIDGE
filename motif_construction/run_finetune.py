#!/usr/bin/env python3
# coding: utf-8
"""
Required CLI flags (others are ignored):
    --model_type           dna           (kept for compatibility, unused)
    --tokenizer_name       path|name
    --model_name_or_path   path|name
    --task_name            dnaprom       (ignored, kept so your bash loop works)
    --do_visualize                          (flag)
    --visualize_data_dir   DIR           (contains *.tsv with seq \t label)
    --visualize_models     K             (k-mer length, int ≥1)
    --data_dir             same as above (kept for compatibility)
    --max_seq_length       INT
    --per_gpu_pred_batch_size INT
    --output_dir           path (unused, kept)
    --predict_dir          DIR           (results will be written here)
    --n_process            INT           (ignored - tokenisation is fast)

Outputs:
    {predict_dir}/atten.npy        - (N, L) normalised attention scores
    {predict_dir}/pred_results.npy - (N,)  probability of positive class
"""
import argparse, os, json, random, numpy as np, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_tsv(path: str, max_len: int):
    """
    Reads a TSV of  <sequence>\t<label>
    Skips blank lines and any header row whose label field is non-numeric.
    """
    seqs, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seq, *lab = line.rstrip("\n").split("\t")

            # ── handle header or bad label ──
            if lab and not lab[0].lstrip("-").isdigit():
                # treat as header row; just skip it
                continue

            seqs.append(seq.upper()[:max_len])
            labels.append(int(lab[0]) if lab else 0)   # default 0 if no label
    return seqs, labels


def build_dataset(tokenizer, seqs, labels, max_len):
    enc = tokenizer(
        seqs,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
        return_tensors="pt",
    )
    lbl = torch.tensor(labels, dtype=torch.long)
    return torch.utils.data.TensorDataset(
        enc["input_ids"], enc["attention_mask"], lbl
    )


def attention_scores(attn, kmer):
    """
    attn: (heads, L, L) tensor from model output
    Returns one vector (L,) after head-sum & k-mer smoothing (as in original).
    """
    # head-sum on CLS row: Σ_h a[h, 0, i]
    row = attn.sum(dim=0)[0]       # shape (L,)
    # slide a k-mer window & diffuse scores like original implementation
    if kmer == 1:
        return row / (row.norm() + 1e-12)
    tmp = torch.zeros_like(row)
    counts = torch.zeros_like(row)
    for i in range(len(row) - kmer + 1):
        w = row[i : i + kmer]
        tmp[i : i + kmer] += w.sum()
        counts[i : i + kmer] += 1
    scores = tmp / counts.clamp_min(1)
    return scores / (scores.norm() + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    # keep only the 13 required args ----------------------------------------- #
    parser.add_argument("--model_type")  # ignored
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--task_name")   # ignored
    parser.add_argument("--do_visualize", action="store_true")
    parser.add_argument("--visualize_data_dir", required=True)
    parser.add_argument("--visualize_models", type=int, default=1)
    parser.add_argument("--data_dir")  # ignored
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--per_gpu_pred_batch_size", type=int, default=8)
    parser.add_argument("--output_dir")  # ignored
    parser.add_argument("--predict_dir", required=True)
    parser.add_argument("--n_process")  # ignored
    args = parser.parse_args()

    assert args.do_visualize, "This script only supports --do_visualize mode."

    # reproducibility -------------------------------------------------------- #
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model + tokenizer ------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, output_attentions=True
    ).to(device).eval()

    # data ------------------------------------------------------------------- #
    tsv_path = os.path.join(args.visualize_data_dir, "dev.tsv")
    seqs, labels = load_tsv(tsv_path, args.max_seq_length)
    dataset = build_dataset(tokenizer, seqs, labels, args.max_seq_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.per_gpu_pred_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # inference -------------------------------------------------------------- #
    all_scores = np.zeros((len(dataset), args.max_seq_length))
    all_probs  = np.zeros(len(dataset))
    softmax = torch.nn.Softmax(dim=1)
    k = args.visualize_models

    with torch.no_grad():
        offset = 0
        for batch in tqdm(loader, desc="Visualising"):
            ids, mask, _ = [t.to(device) for t in batch]
            out = model(input_ids=ids, attention_mask=mask)
            logits, attns = out.logits, out.attentions  # tuple (num_layers, B, heads, L, L)
            # use last layer's attention
            last_attn = attns[-1]                       # shape (B, heads, L, L)

            for b in range(ids.size(0)):
                s = attention_scores(last_attn[b], k)   # tensor (L,)
                all_scores[offset + b] = s.cpu().numpy()

            probs = softmax(logits)[:, 1] if logits.size(-1) == 2 else softmax(logits).max(dim=1).values
            all_probs[offset : offset + ids.size(0)] = probs.cpu().numpy()
            offset += ids.size(0)

    # save ------------------------------------------------------------------- #
    os.makedirs(args.predict_dir, exist_ok=True)
    np.save(os.path.join(args.predict_dir, "atten.npy"), all_scores)
    np.save(os.path.join(args.predict_dir, "pred_results.npy"), all_probs)

    # quick JSON log --------------------------------------------------------- #
    meta = dict(N=len(dataset), kmer=k, max_len=args.max_seq_length)
    with open(os.path.join(args.predict_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved scores → {args.predict_dir}")

if __name__ == "__main__":
    main()
