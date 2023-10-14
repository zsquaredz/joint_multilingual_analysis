#!/bin/bash

# ar bg ca zh hr cs da nl en et fi fr gl de el he hi hu id it ja ko pl pt ro ru sk sl es sv tr uk vi
# pre-process for multilingual models
while read attr; do
  python src/analysis/parafac2.py \
    --treebanks_root ./morphosyntactic_features/data/ud/ud-treebanks-v2.1 \
    --output_dir ./data/analysis_data \
    --langs_to_use ar bg ca zh hr cs da nl en et fi fr gl de el he hi hu id it ja ko pl pt ro ru sk sl es sv tr uk vi \
    --filename xlm-roberta-base-All33-custom-mlm-pretrain \
    --do_preprocess \
    --attribute ${attr}
done < src/analysis/res/properties.lst

# pre-process for monolingual models
while read attr; do
  python src/analysis/parafac2.py \
    --treebanks_root ./morphosyntactic_features/data/ud/ud-treebanks-v2.1 \
    --output_dir ./data/analysis_data \
    --langs_to_use ar bg ca zh hr cs da nl en et fi fr gl de el he hi hu id it ja ko pl pt ro ru sk sl es sv tr uk vi \
    --filename roberta-base-custom-mlm-pretrain \
    --do_preprocess \
    --attribute ${attr}
done < src/analysis/res/properties.lst



