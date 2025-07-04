#!/bin/bash
export PYTHONPATH=/apps/GOT-OCR2.0/GOT_OCR_2_master:$PYTHONPATH
# tokenize_parallel.py 脚本路径
TOKENIZE_SCRIPT="GOT_OCR_2_master/GOT/data/process_single.py"

# tokenizer路径
TOKENIZER_PATH="/data_8t_1/qby/GOT-OCR2_0"

# 定义一个函数，调用tokenize_parallel.py
run_tokenize () {
  local dataset_name=$1
  local images_path=$2
  local annotations_path=$3

  output_json="${annotations_path%.*}_tokenized.json"

  echo "Tokenizing dataset: $dataset_name"
  echo "Images path: $images_path"
  echo "Annotations path: $annotations_path"
  echo "Output path: $output_json"

  python $TOKENIZE_SCRIPT \
    --dataset_name "$dataset_name" \
    --input_json "$annotations_path" \
    --image_path "$images_path" \
    --output_json "$output_json" \
    --tokenizer_name "$TOKENIZER_PATH" \
    --num_workers 12

  echo "Done tokenizing $dataset_name"
  echo "------------------------------"
}

# 调用各个数据集
run_tokenize "data_1" "/data_8t_1/dataset/tfr-dataset/image_and_json/TR/images" "/data_8t_1/dataset/tfr-dataset/image_and_json/TR/clean_data.json"
run_tokenize "data_2" "/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/images" "/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/clean_data.json"
run_tokenize "data_3" "/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/images" "/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/clean_data.json"
run_tokenize "data_4" "/data_8t_1/dataset/tfr-dataset/MFR/val_image_and_json/images" "/data_8t_1/dataset/tfr-dataset/MFR/val_image_and_json/data.json"
run_tokenize "test-longest-gpt-replies" "/apps/GOT-OCR2.0/test_image/output_images" "/apps/GOT-OCR2.0/test_image/longest_gpt_replies_full.json"

echo "All datasets tokenized."
