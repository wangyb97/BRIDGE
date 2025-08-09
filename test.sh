python main.py \
    --train \
    --data_path ./dataset \
    --data_file LIN28B_HEK293 \
    --device_num 0 \
    --early_stopping 20 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model \
    --lr 0.001

python main.py \
    --validate \
    --data_path ./dataset \
    --data_file LIN28B_HEK293 \
    --device_num 0 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model

python main.py \
    --dynamic_predict \
    --data_path ./dataset \
    --data_file AGGF1_K562 \
    --device_num 0 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model

python variant_aware_GWAS.py \
    --gwas_after_variation \
    --fasta_sequence_path ./dataset_variant/AGGF1_K562.fa \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model \
    --variant_out_file ./results/variants/AGGF1_K562_after_mut.txt \
    --device cuda:3
