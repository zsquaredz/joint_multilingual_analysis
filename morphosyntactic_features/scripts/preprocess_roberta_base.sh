#!/bin/bash

# for monolingual models
while read line; do
  CORPUS=($line)
  echo "python preprocess_treebank.py ${CORPUS[0]} --roberta roberta-base"
  python preprocess_treebank.py ${CORPUS[0]} \
  --roberta roberta-base \
  --use-gpu \
  --use_own_lm \
  --exp_name "" \
  --lang ${CORPUS[1]} \
  --model_path ../out/${CORPUS[1]}_pretraining_output/final
done < scripts/languages_all33.lst