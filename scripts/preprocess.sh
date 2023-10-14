#!/bin/bash


# SPLIT=00  # Presplitted for parallel processing
# note that Chinese need special treatment by passing in the flag --chinese

# Arabic Bulgarian Catalan ChineseT Croatian Czech Danish Dutch English Estonian Finnish French
# Galician German Greek Hebrew Hindi Hungarian Indonesian Italian Japanese Korean  
# Polish Portuguese Romanian Russian Slovak Slovenian Spanish Swedish Turkish Ukrainian Vietnamese

for l in Arabic Bulgarian Catalan ChineseT Croatian Czech Danish Dutch English Estonian Finnish French Galician German Greek Hebrew Hindi Hungarian Indonesian Italian Japanese Korean Polish Portuguese Romanian Russian Slovak Slovenian Spanish Swedish Turkish Ukrainian Vietnamese
do
  LANG=$l
  echo "preprocessing data for ${l}"
  DOWNLOAD_DATA_DIR="./data/"
  OUTPUT_DIR="./data/processed"
  mkdir -p $OUTPUT_DIR/$LANG/
  mkdir -p $OUTPUT_DIR/pretraining/${LANG}/

  # Convert data
  python3 src/utils/get_wiki_text.py --input_directory $DOWNLOAD_DATA_DIR/$LANG/ --output_directory $OUTPUT_DIR/$LANG/

  # Preprocess
  HOME=$HOME_DIR TRANSFORMERS_CACHE=${HOME_DIR}/.cache/ python3 src/utils/preprocess.py \
    --source_directory $OUTPUT_DIR/$LANG/ \
    --output_directory $OUTPUT_DIR/pretraining/${LANG}/ \
    --tokenizer_path xlm-roberta-base \
    --max_seq_len 512 \
    --min_load_len 10 \
    --rank 0
done

