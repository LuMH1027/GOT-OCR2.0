import json
import os
import shutil

CONVERSATION_DATA = {
    'data_1': {
        'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/TR/images',
        'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/TR/clean_data.json',
    },
    'data_2': {
        'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/images',
        'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/clean_data.json',
    },
    'data_3': {
        'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/images',
        'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/clean_data.json',
    },
    'data_4': {
        'images': '/data_8t_1/dataset/tfr-dataset/MFR/val_image_and_json/images',
        'annotations': '/data_8t_1/dataset/tfr-dataset/MFR/val_image_and_json/data.json',
    }
}


def extract_and_save_with_images(conversation_data, top_k=1000,
                                 output_json="longest_gpt_replies_full.json",
                                 output_images_dir="./output_images"):
    os.makedirs(output_images_dir, exist_ok=True)

    samples_with_max_gpt_len = []

    # 先遍历所有数据，收集样本和图片根目录
    for key, data_info in conversation_data.items():
        annotation_path = data_info['annotations']
        images_root = data_info['images']
        print(f"Loading {annotation_path} ...")
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            for item in data_list:
                if "conversations" not in item:
                    continue
                max_len = max((len(conv.get("value", "")) for conv in item["conversations"] if conv.get(
                    "from") == "gpt"), default=0)
                # 把图片根目录也一并传入，方便后面复制
                samples_with_max_gpt_len.append((max_len, item, images_root))

    # 排序取前top_k
    samples_with_max_gpt_len.sort(key=lambda x: x[0], reverse=True)
    top_samples = samples_with_max_gpt_len[:top_k]

    # 复制图片和构造输出数据
    output_data = []
    for length, sample, images_root in top_samples:
        output_data.append(sample)
        img_name = sample.get("image", "")
        src_path = os.path.join(images_root, img_name)
        dst_path = os.path.join(output_images_dir, img_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: image file not found: {src_path}")

    # 保存 JSON
    with open(output_json, 'w', encoding='utf-8') as f_out:
        json.dump(output_data, f_out, ensure_ascii=False, indent=2)

    print(f"Saved top {top_k} samples to {output_json}")
    print(f"Copied corresponding images to {output_images_dir}")


if __name__ == "__main__":
    extract_and_save_with_images(CONVERSATION_DATA, top_k=1000)
