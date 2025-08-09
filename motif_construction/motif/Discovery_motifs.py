#!/usr/bin/env python3
"""Motif Discovery Pipeline using Attention Scores and Hypergeometric Filtering"""

from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
import ahocorasick
from operator import itemgetter
from scipy.stats import hypergeom
import statsmodels.stats.multitest as smm
from Bio import Align


def kmer_to_rna(kmers: str) -> str:
    """Reconstruct full RNA string from space-separated kmers."""
    parts = kmers.split(" ")
    rna = "".join([k[0] for k in parts[:-1]] + [parts[-1]])
    assert len(rna) == len(parts) + len(parts[0]) - 1, "Reconstruction failed"
    return rna


def rna_to_kmers(rna: str, k: int) -> str:
    """Convert RNA string to overlapping k-mers with space separators."""
    return " ".join(rna[i: i + k] for i in range(len(rna) - k + 1))


def find_true_regions(condition: np.ndarray, *, min_len: int = 5) -> np.ndarray:
    """Identify contiguous True blocks ≥ min_len in a boolean mask."""
    delta = np.diff(condition)
    changes, = delta.nonzero()
    changes += 1
    if condition[0]:
        changes = np.r_[0, changes]
    if condition[-1]:
        changes = np.r_[changes, condition.size]
    changes.shape = (-1, 2)
    return changes[(changes[:, 1] - changes[:, 0]) >= min_len]


def extract_attention_regions(
    scores: np.ndarray, *, min_len: int = 5, condition: Any | None = None
) -> np.ndarray:
    """Detect high-attention regions using default or custom threshold."""
    if condition is None:
        above_mean = scores > scores.mean()
        above_min = scores > scores.min()
        condition = np.logical_and(above_mean, above_min)
    return find_true_regions(np.asarray(condition), min_len=min_len)


def _count_motif_occurrences(
    rna_list: Sequence[str], motifs: Sequence[str], *, allow_multi: bool = False
) -> Dict[str, int]:
    """Count motif occurrences using Aho-Corasick automaton."""
    automaton = ahocorasick.Automaton()
    counts = {m: 0 for m in motifs}
    for i, motif in enumerate(motifs):
        automaton.add_word(motif, (i, motif))
    automaton.make_automaton()

    for rna in rna_list:
        hits = sorted(map(itemgetter(1), automaton.iter(rna)))
        seen = set()
        for _, motif in hits:
            if allow_multi or motif not in seen:
                counts[motif] += 1
                seen.add(motif)
    return counts


def perform_hypergeometric_test(
    rna_pos: Sequence[str],
    rna_neg: Sequence[str],
    motifs: Sequence[str],
    *,
    p_adjust_method: str = "fdr_bh",
    alpha: float = 0.05,
    verbose: bool = False,
    allow_multi: bool = False
) -> List[float]:
    """Run one-sided hypergeometric test for motif enrichment."""
    total = len(rna_pos) + len(rna_neg)
    K = len(rna_pos)

    total_counts = _count_motif_occurrences(rna_pos + rna_neg, motifs, allow_multi=allow_multi)
    pos_counts = _count_motif_occurrences(rna_pos, motifs, allow_multi=allow_multi)

    pvals = []
    for motif in motifs:
        n = total_counts[motif]
        x = pos_counts[motif]
        p = hypergeom.sf(x - 1, total, K, n)
        if verbose and p < 1e-5:
            print(f"[+] Motif {motif}: N={total}; K={K}; n={n}; x={x}; p={p:.2e}")
        pvals.append(p)

    if p_adjust_method:
        pvals = smm.multipletests(pvals, alpha=alpha, method=p_adjust_method)[1].tolist()
    return pvals


def filter_significant_motifs(
    rna_pos: Sequence[str],
    rna_neg: Sequence[str],
    motifs: Sequence[str],
    *,
    pval_cutoff: float = 0.05,
    return_indices: bool = False,
    **kwargs
):
    """Filter motifs using FDR threshold."""
    pvals = perform_hypergeometric_test(rna_pos, rna_neg, motifs, **kwargs)
    if return_indices:
        return [i for i, p in enumerate(pvals) if p < pval_cutoff]
    return [motifs[i] for i, p in enumerate(pvals) if p < pval_cutoff]


def merge_similar_motifs(
    motif_dict: Dict[str, Dict[str, Any]],
    *,
    min_len: int = 5,
    align_all_ties: bool = True,
    score_threshold: Any | None = None
) -> Dict[str, Dict[str, Any]]:
    """Greedily merge similar motifs using pairwise alignment."""
    aligner = Align.PairwiseAligner()
    aligner.internal_gap_score = -1e4

    merged = {}
    for motif in sorted(motif_dict, key=len):
        if not merged:
            merged[motif] = motif_dict[motif]
            continue

        alignments, keys = [], []
        for ref in merged:
            if motif == ref:
                continue
            aln = aligner.align(motif, ref)[0]
            threshold = score_threshold if score_threshold is not None else max(min_len - 1, 0.5 * min(len(motif), len(ref)))
            if aln.score >= threshold:
                alignments.append(aln)
                keys.append(ref)

        if not alignments:
            merged[motif] = motif_dict[motif]
            continue

        best_score = max(a.score for a in alignments)
        selected = [i for i, a in enumerate(alignments) if a.score == best_score]

        for i in (selected if align_all_ties else selected[:1]):
            aln, ref_key = alignments[i], keys[i]
            l_off = aln.aligned[0][0][0] - aln.aligned[1][0][0]
            r_off = (
                len(motif) - aln.aligned[0][0][1]
                if aln.aligned[0][0][1] <= len(motif) and aln.aligned[1][0][1] == len(ref_key)
                else aln.aligned[1][0][1] - len(ref_key)
                if aln.aligned[0][0][1] == len(motif)
                else len(motif) - aln.aligned[0][0][1]
            )
            merged[ref_key]["rna_indices"].extend(motif_dict[motif]["rna_indices"])
            new_positions = [(s + l_off, e - r_off) for s, e in motif_dict[motif]["attention_regions"]]
            merged[ref_key]["attention_regions"].extend(new_positions)

    return merged


def extract_fixed_windows(
    motif_dict: Dict[str, Dict[str, Any]],
    rna_inputs: Sequence[str],
    *,
    window_size: int = 24
) -> Dict[str, Dict[str, Any]]:
    """Extract fixed-length windows centered on motifs."""
    result = {}
    for motif, info in motif_dict.items():
        result[motif] = {"rna_indices": [], "attention_regions": [], "seqs": []}
        for idx, (start, end) in zip(info["rna_indices"], info["attention_regions"]):
            core_len = end - start
            extra = window_size - core_len
            left = extra // 2
            right = extra - left
            new_start = start - left
            new_end = end + right
            if 0 <= new_start < new_end <= len(rna_inputs[idx]):
                result[motif]["rna_indices"].append(idx)
                result[motif]["attention_regions"].append((new_start, new_end))
                result[motif]["seqs"].append(rna_inputs[idx][new_start:new_end])
    return result


def run_motif_discovery(
    rna_pos: Sequence[str],
    rna_neg: Sequence[str],
    attention_scores: np.ndarray,
    *,
    window_size: int = 24,
    min_len: int = 4,
    pval_cutoff: float = 5e-3,
    min_instances: int = 3,
    align_all_ties: bool = True,
    output_dir: str | None = None,
    verbose: bool = False,
    return_indices: bool = False,
    attention_condition: Any | None = None,
    alignment_condition: Any | None = None
) -> Dict[str, Dict[str, Any]]:
    """Full motif discovery pipeline."""
    if verbose:
        print("\n=== Motif Analysis ===")
        print(f" * Positives: {len(rna_pos)}")
        print(f" * Negatives: {len(rna_neg)}")

    motif_dict: Dict[str, Dict[str, Any]] = {}
    for i, (rna, score) in enumerate(zip(rna_pos, attention_scores)):
        regions = extract_attention_regions(score[:len(rna)], min_len=min_len, condition=attention_condition)
        for start, end in regions:
            motif_seq = rna[start:end]
            motif_dict.setdefault(motif_seq, {"rna_indices": [], "attention_regions": []})
            motif_dict[motif_seq]["rna_indices"].append(i)
            motif_dict[motif_seq]["attention_regions"].append((start, end))

    if verbose:
        print(" * Raw motifs found:", len(motif_dict))
        print(" * Applying statistical filtering …")

    keep = filter_significant_motifs(
        rna_pos, rna_neg, list(motif_dict.keys()),
        pval_cutoff=pval_cutoff, return_indices=return_indices,
        allow_multi=False, verbose=verbose
    )

    if return_indices:
        motif_dict = {k: motif_dict[k] for i, k in enumerate(motif_dict) if i in keep}
    else:
        motif_dict = {k: motif_dict[k] for k in keep}

    motif_dict = merge_similar_motifs(motif_dict, min_len=min_len, align_all_ties=align_all_ties, score_threshold=alignment_condition)
    motif_dict = extract_fixed_windows(motif_dict, rna_pos, window_size=window_size)
    motif_dict = {k: v for k, v in motif_dict.items() if len(v["rna_indices"]) >= min_instances}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for motif, info in motif_dict.items():
            path = os.path.join(output_dir, f"motif_{motif}_{len(info['rna_indices'])}.txt")
            with open(path, "w") as f:
                f.write("\n".join(info["seqs"]))
        if verbose:
            print(f" * Motif sequences saved to → {output_dir}")

    return motif_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Motif Discovery CLI")
    parser.add_argument("--input_rna_dir", required=True, help="Input directory with dev.tsv RNA kmers.")
    parser.add_argument("--attention_dir", required=True, help="Directory with atten.npy and pred_results.npy.")
    parser.add_argument("--motif_length", type=int, default=24, help="Final motif window size.")
    parser.add_argument("--min_region_len", type=int, default=5, help="Minimum contiguous attention region.")
    parser.add_argument("--fdr_cutoff", type=float, default=5e-3, help="Significance threshold for motif filtering.")
    parser.add_argument("--min_motif_count", type=int, default=3, help="Minimum count threshold per motif.")
    parser.add_argument("--align_all_ties", action="store_true", help="Keep all best alignments.")
    parser.add_argument("--output_motif_dir", default=".", help="Directory to save extracted motifs.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--return_idx", action="store_true", help="Return motif indices instead of strings.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    attention = np.load(os.path.join(args.attention_dir, "atten.npy"))
    _ = np.load(os.path.join(args.attention_dir, "pred_results.npy"))
    
    df = pd.read_csv(os.path.join(args.input_rna_dir, "dev.tsv"), sep="\t", header=0, names=["rna_kmers", "label"])
    df["rna"] = df["rna_kmers"].apply(kmer_to_rna)

    df_pos = df[df.label == 1]
    df_neg = df[df.label == 0]
    attention_pos = attention[df_pos.index.values]


    motifs = run_motif_discovery(
        list(df_pos.rna), list(df_neg.rna), attention_pos,
        window_size=args.motif_length,
        min_len=args.min_region_len,
        pval_cutoff=args.fdr_cutoff,
        min_instances=args.min_motif_count,
        output_dir=args.output_motif_dir,
        verbose=args.verbose,
        return_indices=args.return_idx,
    )

    if args.verbose:
        print("\n>>> Final motifs (top):", list(motifs)[:10])


if __name__ == "__main__":
    main()
