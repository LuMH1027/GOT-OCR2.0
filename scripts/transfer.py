from tqdm import tqdm
import lmdb
from PIL import Image
import io
import os
import json
import imghdr
input_paths = ["/data_8t_1/dataset/tfr-dataset/MFR/validation",
               "/data_8t_1/dataset/tfr-dataset/selected_lmdb/MFR",
               "/data_8t_1/dataset/tfr-dataset/selected_lmdb/TFR",]
# LMDB 路径
for lmdb_path in input_paths:

    # 输出路径
    # output_dir = f"{lmdb_path}/image_and_json"
    if "selected_lmdb" in lmdb_path:
        output_dir = lmdb_path.replace("selected_lmdb", "image_and_json")
    else:
        output_dir = lmdb_path.replace("validation", "val_image_and_json")

    image_dir = os.path.join(output_dir, "images")
    json_out = os.path.join(output_dir, "data.json")

    os.makedirs(image_dir, exist_ok=True)

    # 打开 LMDB
    env = lmdb.open(
        lmdb_path,
        max_readers=1,
        readonly=True,
        create=False,
        readahead=False,
        meminit=False,
        lock=False,
    )

    merged_list = []

    with env.begin() as txn:
        num_samples = int(txn.get(b'num-samples').decode())
        print("总样本数:", num_samples)

        for i in tqdm(range(1, num_samples + 1)):
            index = f"{i:09d}"
            key_img = f"image-{index}".encode()
            key_label = f"label-{index}".encode()

            img_bin = txn.get(key_img)
            label_bin = txn.get(key_label)

            if img_bin is None or label_bin is None:
                print(f"[{index}] 缺失图像或标签，跳过")
                continue

            # 判断图像格式
            fmt = imghdr.what(None, h=img_bin)
            if fmt is None:
                fmt = "jpg"  # 默认 fallback
            image_name = f"{index}.{fmt}"

            # 保存图片
            try:
                image = Image.open(io.BytesIO(img_bin))
                image.save(os.path.join(image_dir, image_name))
            except Exception as e:
                print(f"[{index}] 图像保存失败:", e)
                continue

            # 解析标签
            try:
                label_text = label_bin.decode("utf-8", errors="replace")
            except Exception as e:
                print(f"[{index}] 标签解码失败:", e)
                continue

                # 构建 JSON 条目
            merged_list.append({
                "image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nOCR:"
                    },
                    {
                        "from": "gpt",
                        "value": label_text
                    }
                ]
            })

    # 写入合并 JSON
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功导出 {len(merged_list)} 条数据，JSON 写入至: {json_out}")
