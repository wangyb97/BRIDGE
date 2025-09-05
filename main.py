import os
import random
import argparse
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from utils.BRIDGE import BRIDGE

from utils.gen_transformer_embedding import gen_Transformer_embedding, build_Transformer_embeddings
from utils.motif_prior.motif_prior import get_motif_prior_matrix
import torch.utils.data
from transformers import BertModel, BertTokenizer

from utils.train_loop import train, validate, validate2, validate_without_sigmoid
from utils.utils import read_csv, myDataset, myDataset2, param_num, split_dataset, seq2kmer, resolve_dynamic_model_name
from utils.structureFeatures import generateStructureFeatures, read_combined_profile,build_structure_tensor
from utils.FeatureEncoding import dealwithdata
from utils.dataloaders import read_fasta


def log_print(text, color=None, on_color=None, attrs=None):
    try:
        from termcolor import cprint
    except ImportError:
        cprint = None

    try:
        from pycrayon import CrayonClient
    except ImportError:
        CrayonClient = None
        
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def main(args):
    fix_seed(args.seed)
    torch.cuda.set_device(args.device_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = 101
    file_name = args.data_file
    data_path = args.data_path

    if args.train:
        neg_path = os.path.join(data_path, file_name + '_neg.fa')
        pos_path = os.path.join(data_path, file_name + '_pos.fa')
        sequences, structs, label = read_fasta(neg_path, pos_path)
        
        Transformer_emb, attention_weight = build_Transformer_embeddings(
            sequences=list(sequences),
            transformer_path=args.Transformer_path,
            device=device,
            k=1,
            transpose_to_ch_first=True
        )
        structure = build_structure_tensor(structs, max_length)
        biochem = dealwithdata(args.data_file).transpose([0, 2, 1])
        motif = get_motif_prior_matrix(args.data_file)

        [train_emb, train_attn, train_struc, train_motif, train_biochem, train_label], \
        [test_emb, test_attn, test_struc, test_motif, test_biochem, test_label] = split_dataset(
            Transformer_emb,
            attention_weight,
            structure,
            motif,
            biochem,
            label
        )
        train_set = myDataset(train_emb, train_attn, train_struc, train_motif, train_biochem, train_label)
        test_set = myDataset(test_emb, test_attn, test_struc, test_motif, test_biochem, test_label)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32 * 8, shuffle=False)

        model = BRIDGE().to(device)
        # model = torch.nn.DataParallel(model, device_ids=[1, 3])

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
        
        initial_lrate = 0.0016
        drop = 0.8
        epochs_drop = 5.0
        warmup_epochs = 40
        lrs = []
        best_auc = 0
        best_acc = 0
        best_mcc = 0
        best_prc = 0
        best_epoch = 0
        early_stopping = args.early_stopping
        param_num(model)
        model_save_path = './results/model' # args.model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        for epoch in range(1, 201):
            t_met = train(model, device, train_loader, criterion, optimizer, batch_size=32)
            v_met, _, _ = validate(model, device, test_loader, criterion)
            
            if epoch <= warmup_epochs:
                lr = 0.001 * (1.6 * epoch / warmup_epochs)
            else:
                import math
                lr = initial_lrate * math.pow(drop, math.floor((epoch - warmup_epochs) / epochs_drop))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            lrs.append(lr)
            color_best = 'green'
            if best_auc < v_met.auc:
                best_auc = v_met.auc
                best_acc = v_met.acc
                best_mcc = v_met.mcc
                best_prc = v_met.prc
                best_epoch = epoch
                color_best = 'red'
                path_name = os.path.join(model_save_path, file_name+'.pth')
                torch.save(model.state_dict(), path_name)
            if epoch - best_epoch > early_stopping:
                print("Early stop at %d, %s " % (epoch, 'BRIDGE'))
                break
            line = '{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f}, PRC: {:.4f}, MCC: {:.4f}, lr: {:.6f}'.format(
                file_name, epoch, t_met.other[0], t_met.acc, t_met.auc, t_met.prc, t_met.mcc, lr)
            log_print(line, color='green', attrs=['bold'])

            line = '{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} ({:.4f}), PRC: {:.4f}, MCC: {:.4f}, {}'.format(
                file_name, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc, v_met.prc, v_met.mcc, best_epoch)
            log_print(line, color=color_best, attrs=['bold'])
        print("{} auc: {:.4f} acc: {:.4f} prc: {:.4f} mcc: {:.4f}".format(file_name, best_auc, best_acc, best_prc, best_mcc))

    if args.validate:
        fix_seed(args.seed)
        neg_path = os.path.join(data_path, file_name + '_neg.fa')
        pos_path = os.path.join(data_path, file_name + '_pos.fa')
        sequences, structs, label = read_fasta(neg_path, pos_path)
        
        Transformer_emb, attention_weight = build_Transformer_embeddings(
            sequences=list(sequences),
            transformer_path=args.Transformer_path,
            device=device,
            k=1,
            transpose_to_ch_first=True
        )
        structure = build_structure_tensor(structs, max_length)
        biochem = dealwithdata(args.data_file).transpose([0, 2, 1])
        motif = get_motif_prior_matrix(args.data_file)

        [train_emb, train_attn, train_struc, train_motif, train_biochem, train_label], \
        [test_emb, test_attn, test_struc, test_motif, test_biochem, test_label] = split_dataset(
            Transformer_emb,
            attention_weight,
            structure,
            motif,
            biochem,
            label
        )
        test_set = myDataset(test_emb, test_attn, test_struc, test_motif, test_biochem, test_label)
        test_loader = DataLoader(test_set, batch_size=32 * 8, shuffle=False)

        model = BRIDGE().to(device)
        model_file = os.path.join(args.model_save_path, file_name+'.pth')
        if not os.path.exists(model_file):
            print('Model file does not exitsts! Please train first and save the model')
            exit()

        model.load_state_dict(torch.load(model_file))
        model.eval()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        met, y_all, p_all = validate(model, device, test_loader, criterion)
        best_auc = met.auc
        best_acc = met.acc
        best_auprc = met.prc
        best_mcc = met.mcc
        print("{} auc: {:.4f} acc: {:.4f} auprc: {:.4f} mcc: {:.4f}".format(file_name, best_auc, best_acc, best_auprc, best_mcc))

    if args.dynamic_predict:
        fix_seed(args.seed)
        
        model_file = resolve_dynamic_model_name(file_name)
        model_file = os.path.join(args.model_save_path, model_file+'.pth')
        if not os.path.exists(model_file):
            print('Model file does not exitsts! Please train first and save the model')
            exit()

        neg_path = os.path.join(data_path, file_name + '_neg.fa')
        pos_path = os.path.join(data_path, file_name + '_pos.fa')
        sequences, structs, label = read_fasta(neg_path, pos_path)
        
        Transformer_emb, attention_weight = build_Transformer_embeddings(
            sequences=list(sequences),
            transformer_path=args.Transformer_path,
            device=device,
            k=1,
            transpose_to_ch_first=True
        )
        structure = build_structure_tensor(structs, max_length)
        biochem = dealwithdata(args.data_file).transpose([0, 2, 1])
        motif = get_motif_prior_matrix(args.data_file)

        [train_emb, train_attn, train_struc, train_motif, train_biochem, train_label], \
        [test_emb, test_attn, test_struc, test_motif, test_biochem, test_label] = split_dataset(
            Transformer_emb,
            attention_weight,
            structure,
            motif,
            biochem,
            label
        )
        test_set = myDataset(test_emb, test_attn, test_struc, test_motif, test_biochem, test_label)
        test_loader = DataLoader(test_set, batch_size=32 * 8, shuffle=False)

        model = BRIDGE().to(device)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        met, y_all, p_all = validate(model, device, test_loader, criterion)
        best_auc = met.auc
        best_acc = met.acc
        best_auprc = met.prc
        best_mcc = met.mcc
        print("Dynamic prediction mode. {} auc: {:.4f} acc: {:.4f} auprc: {:.4f} mcc: {:.4f}".format(file_name, best_auc, best_acc, best_auprc, best_mcc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to BRIDGE!')
    parser.add_argument('--data_file', default='TIA1_Hela', type=str, help='RBP to train or validate')
    parser.add_argument('--data_path', default='/home/wangyubo/code/BRIDGE/dataset', type=str, help='The data path')
    parser.add_argument('--Transformer_path', default='/home/wangyubo/code/BRIDGE/BERT_Model', type=str, help='BERT model path, in case you have another BERT')
    parser.add_argument('--model_save_path', default='/home/wangyubo/code/BRIDGE/results/model', type=str, help='Save the trained model for dynamic prediction')

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')
    parser.add_argument('--dynamic_predict', default=False, action='store_true')
    
    parser.add_argument('--outdir', default='./results/rsid', type=str, help='Save the output files')
    parser.add_argument('--seed', default=42, type=int, help='The random seed')
    parser.add_argument('--early_stopping', type=int, default=10)
    
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--fasta_sequence_path', required=False, type=str)
    parser.add_argument('--variant_out_file', required=False, type=str)
    
    args = parser.parse_args()
    main(args)
