
nohup deepspeed GOT_OCR_2_master/GOT/train/train_GOT.py \
  --deepspeed GOT_OCR_2_master/zero_config/zero2.json \
  --model_name_or_path /data_8t_1/qby/GOT-OCR2_0/ \
  --seed 42 \
  --use_im_start_end True \
  --bf16 True \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 0.1 \
  --save_total_limit 5 \
  --weight_decay 0. \
  --warmup_ratio 0.001 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --report_to none \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 1 \
  --learning_rate 3e-5 \
  --datasets pdf-ocr \
  --output_dir output > data.log 2>&1 &

pkill -f train_GOT
deepspeed GOT_OCR_2_master/GOT/train/train_GOT_sc.py > training.log 2>&1
# page_ocr
python GOT/eval/evaluate_GOT.py --model-name /data_8t_1/lmh/got/outputs/got_finetune/2025-07-04_18-54-21/checkpoint-31250 --gtfile_path /data_8t_1/dataset/tfr-dataset/image_and_json/80_eval/clean_data.json --image_path  /data_8t_1/dataset/tfr-dataset/image_and_json/80_eval --out_path eval_results/GOT_mathpix_test/page --num-chunks 4 --datatype OCR > eval_finetuned.log 2>&1
python GOT/eval/evaluate_GOT.py --model-name /data_8t_1/qby/GOT-OCR2_0  --gtfile_path /data_8t_1/dataset/tfr-dataset/image_and_json/80_eval/clean_data.json --image_path  /data_8t_1/dataset/tfr-dataset/image_and_json/80_eval --out_path eval_results/GOT_mathpix_test_origin/page --num-chunks 4 --datatype OCR > eval_origin.log 2>&1
# line_ocr
python GOT/eval/evaluate_GOT.py --model-name /data_8t_1/lmh/got/outputs/got_finetune/2025-07-04_18-54-21/checkpoint-31250 --gtfile_path /data_8t_1/dataset/tfr-dataset/image_and_json/80_eval/clean_data.json --image_path  /data_8t_1/dataset/tfr-dataset/image_and_json/80_eval/images --out_path eval_results/GOT_mathpix_test/line --num-chunks 4 --datatype OCR > eval_finetuned.log 2>&1
python GOT/eval/evaluate_GOT.py --model-name /data_8t_1/qby/GOT-OCR2_0  --gtfile_path /data_8t_1/dataset/tfr-dataset/image_and_json/80_eval/clean_data.json --image_path  /data_8t_1/dataset/tfr-dataset/image_and_json/80_eval/images --out_path eval_results/GOT_mathpix_test_origin/line --num-chunks 4 --datatype OCR > eval_origin.log 2>&1

accelerate launch --config_file configs/accelerate_local.yaml GOT_OCR_2_master/GOT/train/train_GOT_sc.py > training.log 2>&1
nohup accelerate launch --config_file configs/accelerate_local.yaml GOT_OCR_2_master/GOT/train/train_GOT_sc.py > training.log 2>&1 &
python GOT_OCR_2_master/GOT/train/train_GOT_sc.py
nohup tensorboard --logdir /data_8t_1/lmh/got/outputs/ --bind_all --port 7272 --window_title TFR_train > /dev/null 2>&1 &

# gradio
cd GOT_OCR_2_master
nohup python test_gradio.py >gradio_output.log 2>&1 &

pkill -f "tensorboard"