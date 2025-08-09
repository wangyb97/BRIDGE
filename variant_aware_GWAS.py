#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

# ---------------------------------------------------------------------------
# Third-party / project-specific utilities (assumed to exist in PYTHONPATH)
# ---------------------------------------------------------------------------
from utils.models.bridge import BridgeModel
from utils.gen_Transformer_embedding import rbpformer_encode_sequences
from utils.train_loop import train, validate, validate2, validate_without_sigmoid
from utils.utils import read_csv, myDataset, myDataset2, param_num, split_dataset, seq2kmer,RBPInferDataset
from utils.FeatureEncoding import dealwithdata,dealwithdata2
from utils.motif_prior.motif_prior import get_motif_prior_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPLEMENT: Dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_fasta(fasta_path: Path) -> Tuple[List[str], List[str]]:
    """Return (names, seqs) from a single-line FASTA file."""
    names, seqs = [], []
    with fasta_path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                names.append(line.rstrip("\n"))
            else:
                seqs.append(line.rstrip("\n").upper())
    return names, seqs


def open_output(out_path: os.PathLike | str) -> Path:
    out_path = Path(out_path)              # ← 统一成 Path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# Variant utilities
# ---------------------------------------------------------------------------
def parse_variant_block(fasta_header: str) -> Tuple[int, str, str, str, int]:
    """
    Parse fasta header and return
        variant_pos, ref_base, alt_base, strand, seq_start
    Header example:
    >variant_1 chr1:27891903-27892003(-)[...]{NA} 27891953:T>A ...
    """
    fields = fasta_header.lstrip(">").split()
    if len(fields) < 3:
        raise ValueError("Unexpected FASTA header format")

    # chr region
    region = fields[1]                             # chr1:27891903-27892003(-)[...]
    strand = region.split("(")[1].split(")")[0]    # + / -
    seq_start = int(region.split(":")[1].split("-")[0])

    # variant
    var_info = fields[-2]                          # 27891953:T>A
    variant_pos = int(var_info.split(":")[0])
    ref_base, alt_base = var_info.split(":")[1].split(">")

    return variant_pos, ref_base, alt_base, strand, seq_start


def apply_complement(base: str) -> str:
    return COMPLEMENT.get(base, base)


def substitute_base(seq: str, pos0: int, alt: str) -> str:
    """Replace seq[pos0] with alt and return new string."""
    if seq[pos0] == alt:
        return seq  # already substituted
    seq_list = list(seq)
    seq_list[pos0] = alt
    return "".join(seq_list)


# ---------------------------------------------------------------------------
# Model loaders (with caching)
# ---------------------------------------------------------------------------
class ModelHub:
    """Caches heavy models & tokenizers to avoid redundant I/O."""

    def __init__(self, transformer_path: Path, device: torch.device) -> None:
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(transformer_path, do_lower_case=False)
        self.transformer = (
            BertModel.from_pretrained(transformer_path).to(device).eval()
        )
        self.bridge_cache: Dict[str, BridgeModel] = {}

    def load_bridge(self, model_dir: Path, filename_stem: str) -> BridgeModel | None:
        """Return cached BRIDGE model or load it from disk."""
        if filename_stem in self.bridge_cache:
            return self.bridge_cache[filename_stem]

        model_file = model_dir / f"{filename_stem}.pth"
        if not model_file.exists():
            logging.warning("Model not found for %s → skip", filename_stem)
            return None

        model = BridgeModel().to(self.device)
        model.load_state_dict(torch.load(model_file, map_location=self.device))
        model.eval()
        self.bridge_cache[filename_stem] = model
        return model


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def process_sequences(
    names: List[str],
    seqs: List[str],
    args: argparse.Namespace,
    hub: ModelHub,
) -> None:
    """Iterate over sequences, perform modification + prediction, append results."""

    out_fp = open_output(Path(args.variant_out_file))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=hub.device))

    with out_fp.open("a") as fout:
        for header, seq in zip(names, seqs):
            try:
                var_pos, ref, alt, strand, seq_start = parse_variant_block(header)
            except ValueError as err:
                logging.error("%s → %s", header, err)
                continue

            # strand-aware base adjustment
            if strand == "-":
                ref, alt = apply_complement(ref), apply_complement(alt)

            # 0-based index of variant in sequence
            idx0 = var_pos - seq_start
            if idx0 < 0 or idx0 >= len(seq):
                logging.error("Variant index out of bounds (%s)", header)
                continue
            if seq[idx0] != ref:
                logging.error("Ref base mismatch (%s) — skip", header)
                continue

            modified_seq = substitute_base(seq, idx0, alt)

            embed, _ = rbpformer_encode_sequences(
                [modified_seq], hub.transformer, hub.tokenizer, hub.device, k=1
            )
            test_emb = embed.transpose([0, 2, 1])
            test_attn = np.zeros((len(seq), 101, 103))              
            struct = np.zeros((1, 1, 101))
            motif = np.zeros((1, 1, 101))
            bio_chem = dealwithdata2(modified_seq).transpose([0, 2, 1])
            
            dataset = RBPInferDataset(
                embedding=test_emb, attn=test_attn, struct=struct,
                motif=motif, biochem=bio_chem
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            filename_stem = Path(args.fasta_sequence_path).stem
            bridge = hub.load_bridge(Path(args.model_save_path), filename_stem)
            if bridge is None:
                continue

            prob = validate_without_sigmoid(bridge, hub.device, loader, criterion).item()
            fout.write(f"{header.lstrip('>')}\tPrediction_score:{prob:.6f}\n")
            # logging.info("Processed %s | score=%.6f", filename_stem, prob)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="after-variation scoring")
    parser.add_argument("--after_variation", action="store_true", help="Run scoring pipeline")
    parser.add_argument("--fasta_sequence_path", required=True, type=Path)
    parser.add_argument("--variant_out_file", required=True)
    parser.add_argument("--Transformer_path", required=True, type=Path)
    parser.add_argument("--model_save_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)

    if not args.after_variation:
        logging.info("--after_variation flag not set; nothing to do.")
        return

    logging.info("Loading FASTA from %s", args.fasta_sequence_path)
    headers, sequences = read_fasta(args.fasta_sequence_path)
    sequences_np = np.array(sequences)

    hub = ModelHub(args.Transformer_path, device)
    process_sequences(headers, sequences_np, args, hub)

    logging.info("Finished. Results appended to %s",
                 args.variant_out_file)


if __name__ == "__main__":
    main()
