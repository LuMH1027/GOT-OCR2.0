import json
from tqdm import tqdm
from transformers import AutoTokenizer
import os
# 指定模型路径和数据路径
model_path = "/data_8t_1/qby/GOT-OCR2_0"
json_paths = [
    "/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/data.json",
    "/data_8t_1/dataset/tfr-dataset/image_and_json/TR/data.json",
    "/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/data.json"
]

# 加载 tokenizer（必须包含自定义特殊 token）
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 获取 im token ID（确保与模型一致）
im_start_token = 151857
im_end_token = 151858
# 遍历每个 json 路径进行清洗
for path in json_paths:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clean_data = []
    bad_data = []

    print(f"\n🧹 Cleaning: {path}")
    for sample in tqdm(data):
        try:
            text = sample["conversations"][1]["value"]
            input_ids = tokenizer(text).input_ids

            if input_ids.count(im_start_token) != input_ids.count(im_end_token):
                bad_data.append(sample)
            else:
                clean_data.append(sample)
        except Exception as e:
            sample["error"] = str(e)
            bad_data.append(sample)

    # 输出 clean/bad 数据
    base_dir = os.path.dirname(path)
    with open(os.path.join(base_dir, "clean_data.json"), "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(base_dir, "bad_data.json"), "w", encoding="utf-8") as f:
        json.dump(bad_data, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Done: {len(clean_data)} valid | ❌ {len(bad_data)} invalid saved.")
