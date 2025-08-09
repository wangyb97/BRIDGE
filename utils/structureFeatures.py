import argparse
import os
import sys
# from keras.preprocessing.sequence import pad_sequences
import re
import linecache
import numpy as np
from functools import reduce
from collections import OrderedDict
from typing import List
import numpy as np

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


def list_to_str(lst):
    ''' Given a list, return the string of that list with tab separators
    '''
    return reduce((lambda s, f: s + '\t' + str(f)), lst, '')


def concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region):
    combine_list = [pairedness.split(), hairpin_loop.split(), internal_loop.split(), multi_loop.split(),
                    external_region.split()]
    return np.array(combine_list).T


def defineExperimentPaths(basic_path, name_id):
    path = basic_path + str(name_id) + '/'
    E_path = basic_path + str(name_id) + '/E/'
    H_path = basic_path + str(name_id) + '/H/'
    I_path = basic_path + str(name_id) + '/I/'
    M_path = basic_path + str(name_id) + '/M/'
    mk_dir(E_path)
    mk_dir(H_path)
    mk_dir(I_path)
    mk_dir(M_path)
    return path, E_path, H_path, I_path, M_path


def read_combined_profile(file_path):
    i = 0
    secondary_structure_list = []
    filelines = linecache.getlines(file_path)
    file_length = len(filelines)
    # print(file_length)
    while i <= file_length - 1:
        pairedness = re.sub('[\s+]', ' ', filelines[i + 1].strip())
        hairpin_loop = re.sub('[\s+]', ' ', filelines[i + 2].strip())
        internal_loop = re.sub('[\s+]', ' ', filelines[i + 3].strip())
        multi_loop = re.sub('[\s+]', ' ', filelines[i + 4].strip())
        external_region = re.sub('[\s+]', ' ', filelines[i + 5].strip())
        combine_array = concatenate(pairedness, hairpin_loop, internal_loop, multi_loop, external_region)
        # X = pad_sequences(combine_array, maxlen=int(160), dtype=np.str, value=seq_encoding_keys.index('UNK'),
        #                   padding='post')
        secondary_structure_list.append(combine_array)
        i = i + 6

    return np.array(secondary_structure_list).astype(float)


def definecombinePaths(basic_path, name_id):
    path = basic_path + str(name_id) + '/'
    E_path = basic_path + str(name_id) + '/E/'
    H_path = basic_path + str(name_id) + '/H/'
    I_path = basic_path + str(name_id) + '/I/'
    M_path = basic_path + str(name_id) + '/M/'
    return path, E_path, H_path, I_path, M_path


def run_RNA(fasta_path, script_path, E_path, H_path, I_path, M_path, W, L, u):
    os.system(
        script_path + '/E_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        E_path + 'E_profile.txt')
    os.system(
        script_path + '/H_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        H_path + 'H_profile.txt')
    os.system(
        script_path + '/I_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        I_path + 'I_profile.txt')
    os.system(
        script_path + '/M_RNAplfold -W ' + str(W) + ' -L ' + str(L) + ' -u ' + str(u) + ' <' + fasta_path + ' ' + '>' +
        M_path + 'M_profile.txt')


def generateStructureFeatures(dataset_path, script_path, basic_path, W, L, u, dataset_name=''):
    path, E_path, H_path, I_path, M_path = defineExperimentPaths(
        basic_path, dataset_name)
    if not os.path.exists(basic_path+'/combined_profile.txt'):
        run_RNA(dataset_path, script_path, E_path, H_path, I_path, M_path, W=W, L=L, u=u)
        fEprofile = open(E_path + 'E_profile.txt')
        Eprofiles = fEprofile.readlines()

        fHprofile = open(H_path + 'H_profile.txt')
        Hprofiles = fHprofile.readlines()

        fIprofile = open(I_path + 'I_profile.txt')
        Iprofiles = fIprofile.readlines()

        fMprofile = open(M_path + 'M_profile.txt')
        Mprofiles = fMprofile.readlines()

        mw = int(1)

        fhout = open(path + 'combined_profile.txt', 'w')

        for i in range(0, int(len(Eprofiles) / 2)):
            id = Eprofiles[i * 2].split()[0]
            print(id, file=fhout)
            E_prob = Eprofiles[i * 2 + 1].split()
            H_prob = Hprofiles[i * 2 + 1].split()
            I_prob = Iprofiles[i * 2 + 1].split()
            M_prob = Mprofiles[i * 2 + 1].split()
            P_prob = list(
                map((lambda a, b, c, d: 1 - float(a) - float(b) - float(c) - float(d)), E_prob, H_prob, I_prob, M_prob))
            print(list_to_str(P_prob[mw - 1:len(P_prob)]), file=fhout)
            print(list_to_str(E_prob[mw - 1:len(P_prob)]), file=fhout)
            print(list_to_str(H_prob[mw - 1:len(P_prob)]), file=fhout)
            print(list_to_str(I_prob[mw - 1:len(P_prob)]), file=fhout)
            print(list_to_str(M_prob[mw - 1:len(P_prob)]), file=fhout)
        fhout.close()

    features = read_combined_profile(path + 'combined_profile.txt')
    return features


def build_structure_tensor(structs: List[str], max_length: int) -> np.ndarray:
    """
    Convert a list of comma-separated structure score strings into a 3D tensor.
    This function is functionally identical to the original inline loop code.

    Args:
        structs (List[str]): List of comma-separated structure score strings.
        max_length (int): Maximum sequence length.

    Returns:
        np.ndarray: Array of shape (N, 1, max_length) containing float32 scores.
    """
    structure = np.zeros((len(structs), 1, max_length))
    for i in range(len(structs)):
        struct = structs[i].split(',')
        ti = [float(t) for t in struct]
        ti = np.array(ti).reshape(1, -1)
        structure[i] = np.concatenate([ti], axis=0)
    return structure
