from typing import List, Tuple
import numpy as np
import re

def read_fasta_with_struct_single(
    path: str,
    label_val: int,
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Read a single FASTA-like file (3 lines per record: header, sequence, structure scores).

    Returns:
        sequences (List[str]): List of sequence strings.
        structs (List[str]): List of comma-separated structure score strings (for downstream `.split(',')`).
        labels (np.ndarray): Label array of shape (N, 1) with float32 values.
    """
    sequences: List[str] = []
    structs: List[str] = []

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(f"{path}: line count is not a multiple of 3")

    for i in range(0, len(lines), 3):
        hdr, seq, struct_str = lines[i], lines[i+1], lines[i+2]
        if not hdr.startswith(">"):
            raise ValueError(f"{path}: invalid header at line {i}")
        if not re.fullmatch(r"[ACGTUN]+", seq):
            raise ValueError(f"{path}: invalid sequence at block {i}")

        arr_len = len(struct_str.split(","))
        if len(seq) != arr_len:
            raise ValueError(f"{path}: length mismatch (seq={len(seq)}, struct={arr_len})")

        sequences.append(seq)
        structs.append(struct_str)

    labels = np.full((len(sequences), 1), label_val, dtype=np.float32)
    return sequences, structs, labels


def read_fasta(
    neg_path: str, 
    pos_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read negative and positive FASTA-like files and return in the same format as original read_csv():
        sequences: np.ndarray (dtype=object) of strings
        structs:   np.ndarray (dtype=object) of comma-separated strings
        labels:    np.ndarray float32 of shape (N, 1)
    """
    seq_neg, struct_neg, label_neg = read_fasta_with_struct_single(neg_path, 0)
    seq_pos, struct_pos, label_pos = read_fasta_with_struct_single(pos_path, 1)

    sequences = np.array(seq_pos + seq_neg, dtype=object)
    structs   = np.array(struct_pos + struct_neg, dtype=object)
    labels    = np.vstack([label_pos, label_neg]).astype(np.float32)

    return sequences, structs, labels
