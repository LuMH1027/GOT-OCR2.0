import json

# 输入输出路径
old_json_path = "/data_8t_1/dataset/tfr-dataset/image_and_json/80_eval/labels.json"
new_json_path = "/data_8t_1/dataset/tfr-dataset/image_and_json/80_eval/clean_data.json"
lmdb = False
# 读取旧 JSON
with open(old_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
new_data = []

if lmdb:
    for item in data:
        new_item = {
            "image": item["image"],
            "conversations": [
                {"from": "human", "value": "<image>\nOCR:"},
                {"from": "gpt", "value": item["conversations"]}
            ]
        }
        new_data.append(new_item)
else:
    for k, v in data.items():
        new_item = {
            "image": k,
            "conversations": [
                {"from": "human", "value": "<image>\nOCR:"},
                {"from": "gpt", "value": v["content"]}
            ]
        }
        new_data.append(new_item)
# 写入新 JSON
with open(new_json_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"✅ 已完成格式转换，输出到：{new_json_path}")
