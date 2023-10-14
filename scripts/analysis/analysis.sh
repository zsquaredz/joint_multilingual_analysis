#!/bin/bash

# All-33
while read attr; do
  python src/analysis/parafac2.py \
    --treebanks_root ./morphosyntactic_features/data/ud/ud-treebanks-v2.1 \
    --output_dir ./data/analysis_data \
    --langs_to_use ar bg ca zh hr cs da nl en et fi fr gl de el he hi hu id it ja ko pl pt ro ru sk sl es sv tr uk vi \
    --filename_exp xlm-roberta-base-All33-custom-mlm-pretrain \
    --filename_ctl roberta-base-custom-mlm-pretrain \
    --exp_name all33 \
    --do_parafac2 \
    --layer 0 \
    --rank 768 \
    --verbose \
    --n_iter_max 100 \
    --attribute ${attr}
done < src/analysis/res/properties.lst

# # Div-10
# while read attr; do
#   python src/analysis/parafac2.py \
#     --treebanks_root ./morphosyntactic_features/data/ud/ud-treebanks-v2.1 \
#     --output_dir ./data/analysis_data \
#     --langs_to_use en ru zh ar hi es tr el fi id \
#     --filename_exp xlm-roberta-base-Div10-custom-mlm-pretrain \
#     --filename_ctl roberta-base-custom-mlm-pretrain \
#     --exp_name div10 \
#     --do_parafac2 \
#     --verbose \
#     --layer 0 \
#     --rank 768 \
#     --attribute ${attr}
# done < src/analysis/res/properties.lst

# # Rel-5
# while read attr; do
#   python src/analysis/parafac2.py \
#     --treebanks_root ./morphosyntactic_features/data/ud/ud-treebanks-v2.1 \
#     --output_dir ./data/analysis_data \
#     --langs_to_use en da de nl sv \
#     --filename_exp xlm-roberta-base-Rel5-custom-mlm-pretrain \
#     --filename_ctl roberta-base-custom-mlm-pretrain \
#     --exp_name rel5 \
#     --do_parafac2 \
#     --layer 12 \
#     --rank 768 \
#     --attribute ${attr}
# done < src/analysis/res/properties.lst



