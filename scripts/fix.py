import json

# 输入输出路径
old_json_path = "/data_8t_1/dataset/tfr-dataset/image_and_json/TR/data.json"
new_json_path = "/data_8t_1/dataset/tfr-dataset/image_and_json/TR/data_gpt_format.json"

# 读取旧 JSON
with open(old_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换格式
new_data = []
for item in data:
    new_item = {
        "image": item["image"],
        "conversations": [
            {"from": "human", "value": "<image>\nOCR:"},
            {"from": "gpt", "value": item["conversations"]}
        ]
    }
    new_data.append(new_item)

# 写入新 JSON
with open(new_json_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"✅ 已完成格式转换，输出到：{new_json_path}")
