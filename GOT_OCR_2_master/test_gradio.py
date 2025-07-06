import time
import torch
import gradio as gr
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer, TextStreamer
import io
import contextlib

# GOT-OCR 模型及工具导入
from GOT.model import GOTQwenForCausalLM
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init, KeywordsStoppingCriteria

# 设置常量
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
IMAGE_TOKEN_LEN = 256

# 模型选择
MODEL_CANDIDATES = {
    "GOT-OCR2_0": "/data_8t_1/qby/GOT-OCR2_0",
    "GOT-OCR2_trained": "/data_8t_1/lmh/got/outputs/got_finetune/2025-07-04_18-54-21/checkpoint-31250"
}

current_model_name = list(MODEL_CANDIDATES.keys())[0]
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化模型


def load_model(model_path):
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map=device,
        use_safetensors=True,
        pad_token_id=151643
    ).eval().to(dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    return model, tokenizer


model_path = MODEL_CANDIDATES[current_model_name]
model, tokenizer = load_model(model_path)
image_processor = BlipImageEvalProcessor(image_size=1024)

# GOT-OCR 推理函数


def run_got_ocr(image_path: str) -> tuple[str, str]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor(image).unsqueeze(0).half().to(device)
    image_tensor_high = image_processor(image).unsqueeze(0).half().to(device)

    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * \
        IMAGE_TOKEN_LEN + DEFAULT_IM_END_TOKEN + '\n' + "OCR with format: "
    conv = conv_templates["mpt"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria(
        [stop_str], tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True,
                            skip_special_tokens=True)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        start = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
            output_ids = model.generate(
                input_ids,
                images=[(image_tensor, image_tensor_high)],
                do_sample=False,
                num_beams=1,
                max_new_tokens=8192,
                stopping_criteria=[stopping_criteria],
                no_repeat_ngram_size=20,
                streamer=streamer,
            )
        duration = time.time() - start

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()
    print(f"time: {duration:.2f} seconds")
    return outputs, f"\u8017\u65f6\uff1a{duration:.2f} \u79d2"


# Gradio UI 搭建
css = """
.mdmd { padding: 10px; }
.katex-error { color: inherit; }
"""

with gr.Blocks(css=css, title="GOT-OCR 演示网站") as app:
    gr.Markdown("# GOT-OCR 在线演示 (本地模型运行)")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                label="选择模型",
                choices=list(MODEL_CANDIDATES.keys()),
                value=current_model_name
            )
            input_file = gr.File(
                label="上传图片文件（JPG/PNG）",
                file_types=[".png", ".jpg", ".jpeg"]
            )
            btn_run = gr.Button("开始识别")

            image_preview = gr.Image(label="图片预览", visible=False)
            time_display = gr.Textbox(label="推理耗时", interactive=False)

        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Markdown 结果预览"):
                    md = gr.Markdown(
                        value="等待处理...",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$", "right": "$", "display": False},
                            {"left": "\\[", "right": "\\]", "display": True},
                            {"left": "\\(", "right": "\\)", "display": False}
                        ],
                        elem_classes="mdmd",
                        header_links=False
                    )
                with gr.Tab("Markdown 原始文本"):
                    md_txt = gr.Code(
                        label="Markdown源码",
                        interactive=True,
                        language="markdown",
                        wrap_lines=True
                    )

    def preview_img(file_path):
        return gr.update(value=file_path, visible=True)

    def infer(file_path, model_name):
        global model, tokenizer
        new_path = MODEL_CANDIDATES[model_name]
        if new_path != model_path:
            model, tokenizer = load_model(new_path)
        print(f"Using model: {new_path}")
        if file_path is None:
            raise gr.Error("请上传图像文件")
        result, duration_text = run_got_ocr(file_path)
        return result, result, duration_text

    model_selector.change(lambda name: None, inputs=model_selector)
    input_file.change(preview_img, inputs=input_file, outputs=image_preview)
    btn_run.click(infer, inputs=[input_file, model_selector], outputs=[
                  md, md_txt, time_display])

if __name__ == "__main__":
    app.launch(server_port=7850, server_name="0.0.0.0")
