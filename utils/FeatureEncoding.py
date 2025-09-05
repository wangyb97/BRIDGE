import numpy as np
import collections

def get_1_trids():
    """Return an index mapping for all 1-mers (4 possibilities)."""
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**1
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index   


def get_2_trids():
    """Return an index mapping for all 2-mers (4^2 = 16 combinations)."""
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index    


def get_3_trids():
    """Return an index mapping for all 3-mers (4^3 = 64 combinations)."""
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]        
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index  


def frequency(seq,kmer,coden_dict):
    """Count occurrences of each k-mer along *seq* and return as index→freq.

    Parameters
    ----------
    seq : str
        Input nucleotide sequence (T is internally converted to U).
    kmer : int
        Size of the k-mer (1, 2 or 3).
    code_dict : dict[str, int]
        Mapping from k-mer string to integer index (from get_*_trids).
    """
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i+k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict


def coden(seq,kmer,tris):
    """Return *positional* k-mer frequency tensor (101 × |code_dict|).

    Each position *i* stores the *global* frequency of the k-mer starting at
    *i*. This matches the feature design used in the original BRIDGE paper.
    """
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i+kmer].replace('T', 'U')]]
        vectors[i][coden_dict[seq[i:i+kmer].replace('T', 'U')]] = value/100
    return vectors


def processFastaFile(seq):
    """One-hot encode *seq* (A/U/C/G) with zero-padding up to length 101.

    The last (third) channel is set to 1 for padded positions so the model can
    distinguish real vs. padded residues.
    """
    phys_dic = {
    'A': [1,1,1],
    'U': [0,0,1],
    'C': [0,1,0],
    'G': [1,0,0]} 
    seqLength = len(seq)
    sequence_vector = np.zeros([101, 3])
    for i in range(0, seqLength):
        sequence_vector[i, 0:3] = phys_dic[seq[i]]
    for i in range(seqLength, 101):
        sequence_vector[i, -1] = 1
    return sequence_vector
    
def dpcp(seq):
    """Encode dinucleotide physicochemical properties as 11-dim vectors."""
    phys_dic = {
    #Shift Slide Rise Tilt Roll Twist Stacking_energy Enthalpy Entropy Free_energy Hydrophilicity
    'AA': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
    'AU': [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
    'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24,  0.14,],
    'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14,  -7.6, -19.2, -2.08, 0.08],
    'UA': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
    'UU': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
    'UC': [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
    'UG': [-0.01, -1.78, 3.32,  0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26,  0.17],
    'CA': [ 0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
    'CU': [-0.04, -1.5, 3.3,  0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
    'CC': [ -0.01, -1.78, 3.32, 0.3,  8.7, 32, -11.1, -12.2, -29.7, -3.26,  0.49],
    'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
    'GA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
    'GU': [-0.08, -1.27, 3.18, -0.8,  7,  31, -13.7, -6.6, -18.4, -0.93, 0.44],
    'GC': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
    'GG': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34 ]}
    
    seqLength = len(seq)
    sequence_vector = np.zeros([101, 11])
    k = 2
    for i in range(0, seqLength-1):
        sequence_vector[i, 0:11] = phys_dic[seq[i:i+k]]
    return sequence_vector
        

def nd(seq, seq_length):
    """Cumulative nucleotide density at each position (length = seq_length)."""
    seq = seq.strip()
    nd_list = [None] * seq_length
    for j in range(seq_length):
        #print(seq[0:j])
        if seq[j] == 'A':
            nd_list[j] = round(seq[0:j+1].count('A') / (j + 1), 3)
        elif seq[j] == 'U':
            nd_list[j] = round(seq[0:j+1].count('U') / (j + 1), 3)
        elif seq[j] == 'C':
            nd_list[j] = round(seq[0:j+1].count('C') / (j + 1), 3)
        elif seq[j] == 'G':
            nd_list[j] = round(seq[0:j+1].count('G') / (j + 1), 3)     
    return np.array(nd_list)


def dealwithdata(protein):
    """Batch-encode all sequences from {protein}_pos.fa and {protein}_neg.fa in ./dataset/.

    Parameters
    ----------
    protein : str
    """
    seq_length = 101
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    dataX = []

    for label in ['pos', 'neg']:
        fasta_path = f'./dataset/{protein}_{label}.fa'
        with open(fasta_path, 'r') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 3):
            seq_line = lines[i+1].strip()
            seq_line = seq_line.replace('T', 'U').replace('N', 'A')

            probMatr = processFastaFile(seq_line)
            probMatr_ND = nd(seq_line, seq_length)
            probMatr_DPCP = dpcp(seq_line) / 101

            probMatr_NDCP = np.column_stack((probMatr, probMatr_ND))
            probMatr_NDPCP = np.column_stack((probMatr_NDCP, probMatr_DPCP))

            kmer1 = coden(seq_line, 1, tris1)
            kmer2 = coden(seq_line, 2, tris2)
            kmer3 = coden(seq_line, 3, tris3)
            Kmer = np.hstack((kmer1, kmer2, kmer3))

            Feature_Encoding = np.column_stack((probMatr_NDPCP, Kmer))
            dataX.append(Feature_Encoding)

    dataX = np.stack(dataX, axis=0)
    print(f"[INFO] Encoded {dataX.shape[0]} sequences for {protein}, shape: {dataX.shape}")
    return dataX



def dealwithdata2(seq):
    line = seq
    seq_length = 101
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    # tris4 = get_4_trids()
    dataX = []
    dataY = []
    line = line.replace('T', 'U').replace('N', 'A').strip()
    probMatr = processFastaFile(line)
    probMatr_ND = nd(line,seq_length)
    probMatr_NDCP = np.column_stack((probMatr,probMatr_ND))
    probMatr_DPCP = dpcp(line)/101
    probMatr_NDPCP = np.column_stack((probMatr_NDCP,probMatr_DPCP))
    kmer1 = coden(line.strip(),1,tris1)
    kmer2 = coden(line.strip(),2,tris2) 
    kmer3 = coden(line.strip(),3,tris3) # (101, 64)
    Kmer = np.hstack((kmer1,kmer2,kmer3)) # (101, 84)
    Feature_Encoding = np.column_stack((probMatr_NDPCP,Kmer))  # (101, 99)
    dataX.append(Feature_Encoding.tolist())
    dataX = np.array(dataX)
    
    return dataX # (1, 101, 99)
