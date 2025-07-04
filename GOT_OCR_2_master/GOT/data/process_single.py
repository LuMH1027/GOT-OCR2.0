import os
import json
import copy
import argparse
from multiprocessing import Pool, cpu_count
from megfile import smart_glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from natsort import natsorted
from transformers import AutoTokenizer
from conversation_dataset_qwen import ConversationDataset
import argparse
from datetime import datetime
import os
from GOT.model.vision_encoder.vary_b import build_vary_vit_b
from GOT.utils.utils import smart_tokenizer_and_embedding_resize
from GOT.utils.constants import *
from GOT.utils.arguments import *
from GOT.model import *
import transformers
from pathlib import Path
import torch
from omegaconf import OmegaConf
# 全局变量，子进程中初始化
tokenizer = None
dataset_dummy = None


parser = argparse.ArgumentParser()
parser.add_argument(
    "--configs",
    nargs="*",
    default=["configs/got.yaml"],
    help="Path to the config file",
)
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Used for distributed training')  # ✅ 添加这一行
parser.add_argument("--dataset_name", type=str,
                    default="GOT-OCR2.0", help="数据集名称")
parser.add_argument("--input_json", type=str,
                    required=True, help="原始json路径")
parser.add_argument("--image_path", type=str,
                    required=True, help="对应图片根目录")
parser.add_argument("--output_json", type=str,
                    required=True, help="输出tokenized的json路径")
parser.add_argument("--tokenizer_name", type=str,
                    default="/data_8t_1/qby/GOT-OCR2_0", help="tokenizer路径")
parser.add_argument("--num_workers", type=int,
                    default=cpu_count() - 1, help="并行进程数")
# args = parser.parse_args()
args = parser.parse_args()

config_list = [OmegaConf.load(c) for c in args.configs]
config = OmegaConf.merge(*config_list)
# model_args, data_args, training_args = parser.parse_yaml_file(
#     "configs/got.yaml")
# config = OmegaConf.load("configs/got.yaml")
# 分别提取字段构造 dataclass
model_args = ModelArguments(
    **{k: v for k, v in config.items() if k in ModelArguments.__dataclass_fields__}
)
data_args = DataArguments(
    **{k: v for k, v in config.items() if k in DataArguments.__dataclass_fields__}
)
training_args = TrainingArguments(
    **{k: v for k, v in config.items() if k in TrainingArguments.__dataclass_fields__}
)
print(training_args.per_device_train_batch_size)
# model_args, data_args, training_args = parser.parse_args_into_dataclasses()
save_dir = f"outputs/{model_args.experiment_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
Path(save_dir).mkdir(parents=True, exist_ok=True)

training_args.output_dir = save_dir
OmegaConf.save(config, f"{save_dir}/config.yaml")
pretrain_model = model_args.model_name_or_path
tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrain_model, trust_remote_code=True, padding_side="right", model_max_length=training_args.model_max_length,)

model = GOTQwenForCausalLM.from_pretrained(
    pretrain_model, use_safetensors=True)

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=dict(pad_token='<|endoftext|>'),
    tokenizer=tokenizer,
    model=model,
)

dtype = torch.float32
if training_args.fp16:
    dtype = torch.float16
if training_args.bf16:
    dtype = torch.bfloat16

vision_tower_dict = model.get_model().initialize_vision_modules(
    vision_tower=model_args.vision_tower,
    pretrained_stage1_model=model_args.pretrained_stage1_model,
    freeze_vision_tower=model_args.freeze_vision_tower,
    use_im_start_end=model_args.use_im_start_end,
    vision_select_layer=model_args.vision_select_layer,
    dtype=dtype,
    device=training_args.device
)

model.initialize_vision_tokenizer(
    tokenizer=tokenizer,
    freeze_lm_model=model_args.freeze_lm_model,
    pretrained_stage1_model=model_args.pretrained_stage1_model,
    device=training_args.device,
)

model.to(dtype=dtype, device=training_args.device)
# 'image_processor_high
# data_args.image_token_len = vision_tower_dict['image_token_len']
data_args.image_token_len = 256
data_args.image_processor = vision_tower_dict['image_processor']
data_args.image_processor_high = vision_tower_dict['image_processor_high']
data_args.use_im_start_end = model_args.use_im_start_end


def init_worker(tokenizer_name, multimodal_cfg, dataset_name):
    global tokenizer
    global dataset_dummy
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, padding_side="right", model_max_length=8192)
    dataset_dummy = ConversationDataset(
        tokenizer=tokenizer,
        datasets=dataset_name,  # 空，因为我们手动传数据
        multimodal_cfg=dict(
            sep_image_conv_front=data_args.sep_image_conv_front,
            image_token_len=data_args.image_token_len,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=data_args.use_im_start_end,
            image_processor=data_args.image_processor,
            image_processor_high=data_args.image_processor_high,
            box_limit=data_args.box_limit,
        )
    )


def process_single_item(args):
    i, data, image_path = args
    # print(f"Processing item {i}...")
    if isinstance(data, dict):
        image_list = []
        image_high_list = []
        flag_num_patches = 1
        if 'image' in data:
            # print("image in data")
            image_path = dataset_dummy.list_image_path[i]
            image_file = data['image']

            # multi-crop or multi page, only support .png files
            # if ('.jpg' not in image_file and '.png' not in image_file and '.jpeg' not in image_file) and ('.jpg' not in image_path and '.png' not in image_path and '.jpeg' not in image_path):
            #     print(
            #         f'image {image_file} is not a valid image file, we thus select 0-th sample instead!')
            #     if image_file[0] == '/':
            #         patch_dir = image_path[:-1] + image_file
            #         patches = smart_glob(patch_dir + '*.png')
            #     else:
            #         patch_dir = image_path + image_file
            #         patches = smart_glob(patch_dir + '*.png')

            #     # print(patches)
            #     if not patches:
            #         print(f'cannot glob the dir {patch_dir}.')
            #         return dataset_dummy.__getitem__(0)

            #     # sort multi images by name
            #     patches = natsorted(patches)
            #     flag_num_patches = len(patches)

            #     for patch in patches:
            #         try:
            #             image = Image.open(patch).convert('RGB')
            #         except:
            #             print(f'cannot identify image file {patch}.')
            #             return dataset_dummy.__getitem__(0)

            #         try:
            #             img = dataset_dummy.image_processor(image)
            #             image_list.append(img)
            #             image_high_list.append(img)
            #         except:
            #             print(
            #                 f'image {image_path + image_file + patch} are broken or grayscale! we thus select 0-th sample instead!')
            #             return dataset_dummy.__getitem__(0)

            # else:
            #     print("image_file", image_file)
            #     flag_num_patches = 1
            #     try:
            #         image = Image.open(
            #             image_path + "/" + image_file).convert('RGB')
            #         print(image.size)
            #         image_array = np.array(image)

            #         print("图像数组形状:", image_array.shape)  # (H, W, 3)
            #     except:
            #         print(
            #             f'cannot identify image file {image_path + image_file}.')
            #         return dataset_dummy.__getitem__(0)
            #     print("2_image_file", image_file)
            #     try:
            #         print("image_processor")
            #         image = dataset_dummy.image_processor(image)
            #     except:
            #         print(
            #             f'image {image_file} are broken or grayscale! we thus select 0-th sample instead!')
            #         return dataset_dummy.__getitem__(0)
            #     print("3_image_file", image_file)
        # print("multimodal_processor")
        conversations = dataset_dummy.multimodal_processor(
            [data["conversations"]], flag_num_patches)
        # print(conversations)
        # exit()
    else:
        # print("data is not a dict, using dummy data")
        conversations = [data]

    image_name = image_path + image_file
    # print(f"Conversations length: {len(conversations)}")
    tokenized = dataset_dummy.token_processor(
        conversations, image_name)
    tokenized_data = {
        "input_ids": tokenized["input_ids"][0].tolist(),
        "labels": tokenized["labels"][0].tolist(),
    }
    new_data = copy.deepcopy(data)
    new_data["tokenized"] = tokenized_data
    return i, new_data


def main():

    # multimodal_cfg配置示例，按需修改
    multimodal_cfg = dict(
        sep_image_conv_front=True,
        image_token_len=256,
        image_aspect_ratio=1.0,
        use_im_start_end=True,
        image_processor=None,
        image_processor_high=None,
        box_limit=512,
    )

    print(f"Loading data from {args.input_json} ...")
    with open(args.input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(
        f"Tokenizing {len(raw_data)} samples with {args.num_workers} workers ...")
    args_list = [(i, raw_data[i], args.image_path)
                 for i in range(len(raw_data))]

    with Pool(processes=args.num_workers, initializer=init_worker, initargs=(args.tokenizer_name, multimodal_cfg, args.dataset_name)) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_single_item, args_list), total=len(raw_data)))

    # 按索引排序恢复原顺序
    results.sort(key=lambda x: x[0])
    tokenized_data = [item[1] for item in results]

    print(f"Saving tokenized data to {args.output_json} ...")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(tokenized_data, f, ensure_ascii=False, indent=2)
    pt_path = args.output_json.replace('.json', '.pt')
    with open(args.output_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    torch.save(data, pt_path)
    print("Saved to tokenized_data.pt ✅")

    print("Done.")


if __name__ == "__main__":
    main()
