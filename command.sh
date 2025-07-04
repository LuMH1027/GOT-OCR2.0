
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
python3 GOT/eval/evaluate_GOT.py --model-name /apps/GOT-OCR2.0/output/checkpoint-14067 --gtfile_path /data_8t_1/qby/Fox_benchmark_data/focus_benchmark_test/cn_page_ocr.json --image_path  /data_8t_1/qby/Fox_benchmark_data/focus_benchmark_test/cn_pdf_png --out_path eval_results/GOT_mathpix_test/ --num-chunks 4 --datatype OCR
python3 GOT/eval/evaluate_GOT.py --model-name /data_8t_1/qby/GOT-OCR2_0 --gtfile_path /data_8t_1/qby/Fox_benchmark_data/focus_benchmark_test/cn_page_ocr.json --image_path  /data_8t_1/qby/Fox_benchmark_data/focus_benchmark_test/cn_pdf_png --out_path eval_results/GOT_mathpix_test_origin/ --num-chunks 4 --datatype OCR
accelerate launch --config_file configs/accelerate_local.yaml GOT_OCR_2_master/GOT/train/train_GOT_sc.py > training.log 2>&1
nohup accelerate launch --config_file configs/accelerate_local.yaml GOT_OCR_2_master/GOT/train/train_GOT_sc.py > training.log 2>&1 &
python GOT_OCR_2_master/GOT/train/train_GOT_sc.py
nohup tensorboard --logdir outputs/ --bind_all --port 7272 --window_title TFR_train > /dev/null 2>&1 &

pkill -f "tensorboard"