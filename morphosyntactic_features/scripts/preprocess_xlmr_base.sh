#!/bin/bash

# for multilingual models All-33
while read line; do
  CORPUS=($line)
  echo "python preprocess_treebank.py ${CORPUS[0]} --xlmr xlm-roberta-base"
  python preprocess_treebank.py ${CORPUS[0]} \
  --xlmr xlm-roberta-base \
  --use-gpu \
  --use_own_lm \
  --exp_name All33 \
  --lang ${CORPUS[1]} \
  --model_path ../out/All33_pretraining_output/final
done < scripts/languages_all33.lst

# # for multilingual models Div-10
# while read line; do
#   CORPUS=($line)
#   echo "python preprocess_treebank.py ${CORPUS[0]} --xlmr xlm-roberta-base"
#   python preprocess_treebank.py ${CORPUS[0]} \
#   --xlmr xlm-roberta-base \
#   --use-gpu \
#   --use_own_lm \
#   --exp_name Div10 \
#   --model_path ../out/Div10_pretraining_output/final
# done < scripts/languages_div10.lst

# # for multilingual models Rel-5
# while read line; do
#   CORPUS=($line)
#   echo "python preprocess_treebank.py ${CORPUS[0]} --xlmr xlm-roberta-base"
#   python preprocess_treebank.py ${CORPUS[0]} \
#   --xlmr xlm-roberta-base \
#   --use-gpu \
#   --use_own_lm \
#   --exp_name Rel5 \
#   --model_path ../out/Rel5_pretraining_output/final
# done < scripts/languages_rel5.lst

