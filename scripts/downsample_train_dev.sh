#!/bin/bash

DATA_DIR="./data/processed/pretraining"
mkdir ${DATA_DIR}/downsampled_dev/
mkdir ${DATA_DIR}/downsampled_train/

python3 src/utils/downsample.py \
  --dirs $DATA_DIR/Arabic/ $DATA_DIR/Bulgarian/ $DATA_DIR/Catalan/ $DATA_DIR/ChineseT/ $DATA_DIR/Croatian/ $DATA_DIR/Czech/ \
        $DATA_DIR/Danish/ $DATA_DIR/Dutch/ $DATA_DIR/English/ $DATA_DIR/Estonian/ $DATA_DIR/Finnish/ $DATA_DIR/French/ \
        $DATA_DIR/Galician/ $DATA_DIR/German/ $DATA_DIR/Greek/ $DATA_DIR/Hebrew/ $DATA_DIR/Hindi/ $DATA_DIR/Hungarian/ \
        $DATA_DIR/Indonesian/ $DATA_DIR/Italian/ $DATA_DIR/Japanese/ $DATA_DIR/Korean/  $DATA_DIR/Polish/ $DATA_DIR/Portuguese/ \
        $DATA_DIR/Romanian/ $DATA_DIR/Russian/ $DATA_DIR/Slovak/ $DATA_DIR/Slovenian/ $DATA_DIR/Spanish/ $DATA_DIR/Swedish/ \
        $DATA_DIR/Turkish/ $DATA_DIR/Ukrainian/ $DATA_DIR/Vietnamese/ \
  --output_dir ${DATA_DIR}/downsampled_train/ \
  --num_examples 70380 \
  --dev_output_dir ${DATA_DIR}/downsampled_dev/

# 72315