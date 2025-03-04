import argparse  # 新增导入
import datetime
import os
import re
import sys

sys.path.append("third_party/Matcha-TTS")
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B", load_jit=False, load_trt=False, fp16=False
)


# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)


# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '雄大的反应十分平静，声称那些只不过是谣言。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
# for i, j in enumerate(
#     cosyvoice.inference_instruct2(
#         "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
#         "用开心的语气说这句话",
#         prompt_speech_16k,
#         stream=False,
#     )
# ):
#     torchaudio.save("instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate)

# # bistream usage, you can use generator as input, this is useful when using text llm model as input
# # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def load_role_config(config_path):
    """加载角色配置文件"""
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 预加载所有提示音频
    for role in config.values():
        try:
            role["prompt_speech"] = load_wav(role["prompt_path"], 16000)
        except Exception as e:
            raise ValueError(f"加载角色音频失败 [{role['prompt_path']}]: {str(e)}")
    return config


def sanitize_filename(text):
    """生成合法文件名（保留10个有效字符）"""
    # 移除特殊字符
    clean_text = re.sub(r'[\\/*?:"<>|()]', "", text)
    # 截取前10个字符
    clean_text = clean_text[:10].strip()
    # 如果处理后为空，使用时间戳
    if not clean_text:
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return clean_text


def get_unique_filename(directory, base_name, extension, line_num=None):
    """生成带有序号的不重复文件名"""
    counter = 1
    while True:
        # 构建文件名
        if line_num is not None:
            main_part = f"{base_name}_{line_num}"
        else:
            main_part = f"{base_name}_combined"

        # 添加序号（第一次不添加）
        if counter > 1:
            full_name = f"{main_part}_{counter}.{extension}"
        else:
            full_name = f"{main_part}.{extension}"

        path = os.path.join(directory, full_name)

        # 检查文件是否存在
        if not os.path.exists(path):
            return path
        counter += 1


def process_txt_file(
    file_path,
    config_path,
    output_dir="output",  # 新增输出目录参数
    merge_files=True,
    role_override=None,
    method="instruct",
):
    # 验证方法参数有效性
    if method not in ["instruct", "zero_shot"]:
        raise ValueError(f"不支持的合成方法: {method}")

    # 加载角色配置
    role_config = load_role_config(config_path)

    # 新增：预检查覆盖角色是否存在
    if role_override:
        if role_override not in role_config:
            raise ValueError(f"覆盖角色不存在于配置中: {role_override}")
        override_config = role_config[role_override]

    all_audio = []
    sample_rate = cosyvoice.sample_rate

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取基础文件名
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = next((l for l in f if l.strip()), "").strip()
    base_name = sanitize_filename(re.sub(r"^\([^)]*\)", "", first_line))  # 移除角色标记

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line for line in f]

    for line_num, line in enumerate(lines):
        text = line.strip()

        # 处理空行（保持原有逻辑）
        if not text:
            if merge_files:
                pause_samples = int(0.8 * sample_rate)
                all_audio.append(torch.zeros(1, pause_samples))
            continue

        # 修改后的角色解析逻辑
        emotion = None
        if role_override:  # 使用覆盖角色
            role = role_override
            # 解析情绪（支持 (情绪) 或 (|情绪) 格式）
            if line.startswith("("):
                match = re.match(r"\([^|)]*(?:\|([^)]+))?\)", line)
                if match:
                    emotion = match.group(1).strip() if match.group(1) else None
                    text = text[match.end() :].strip()
        else:  # 正常解析模式
            role = None
            if line.startswith("("):
                # 匹配整个括号内容
                match = re.match(r"\(([^)]+)\)", line)
                if match:
                    content = match.group(1)
                    # 分割角色和情绪
                    if "|" in content:
                        role_part, emotion_part = content.split("|", 1)
                        role = role_part.strip()
                        emotion = emotion_part.strip()
                    else:
                        # 没有竖线时，整个内容作为情绪
                        emotion = content.strip()
                    text = text[match.end() :].strip()

        # 获取角色配置（新增默认角色逻辑）
        if role_override:
            config = override_config
        else:
            # 当未指定角色时，使用配置中的第一个角色
            if not role:
                first_role = next(iter(role_config.keys()), None)
                if not first_role:
                    raise ValueError(
                        f"第 {line_num + 1} 行缺少角色标识且配置文件中无可用角色"
                    )
                role = first_role
            if role not in role_config:
                raise ValueError(f"未定义的角色: {role} (第 {line_num + 1} 行)")
            config = role_config[role]

        emotion = emotion or config["default_emotion"]

        # 统一使用参数指定的方法
        if method == "instruct":
            instruction = (
                f"用{emotion}的语气说这句话" if emotion else "用自然的语气说这句话"
            )
            generator = cosyvoice.inference_instruct2(
                text, instruction, config["prompt_speech"], stream=False
            )
        elif method == "zero_shot":
            generator = cosyvoice.inference_zero_shot(
                text,
                os.path.splitext(os.path.basename(config["prompt_path"]))[0],
                config["prompt_speech"],
                stream=False,
            )

        for i, j in enumerate(generator):
            audio_data = j["tts_speech"]
            if merge_files:
                all_audio.append(audio_data)

            # 保存独立文件逻辑
            if not merge_files:
                filename = get_unique_filename(
                    output_dir,
                    base_name,
                    "wav",
                    line_num=line_num + 1,  # 传入行号参数
                )
                torchaudio.save(filename, audio_data, sample_rate)
                print(f"生成文件: {filename}")

    # 保存合并文件逻辑
    if merge_files and all_audio:
        combined = torch.cat(all_audio, dim=1)
        output_path = get_unique_filename(output_dir, base_name, "wav")
        torchaudio.save(output_path, combined, sample_rate)
        print(f"合并文件已保存至: {output_path}")


def process_text_input(
    text,
    config_path,
    output_dir="output",
    merge_files=True,
    role_override=None,
    method="instruct",
):
    """处理直接输入的文本"""
    # 加载角色配置
    role_config = load_role_config(config_path)

    # 预检查覆盖角色是否存在
    if role_override:
        if role_override not in role_config:
            raise ValueError(f"覆盖角色不存在于配置中: {role_override}")
        override_config = role_config[role_override]

    all_audio = []
    sample_rate = cosyvoice.sample_rate

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 分割文本为行
    lines = text.strip().split("\n")

    # 获取基础文件名（使用第一行的内容）
    first_line = next((l for l in lines if l.strip()), "").strip()
    base_name = sanitize_filename(re.sub(r"^\([^)]*\)", "", first_line))

    # 处理每一行文本
    for line_num, line in enumerate(lines):
        text = line.strip()

        if not text:
            if merge_files:
                pause_samples = int(0.8 * sample_rate)
                all_audio.append(torch.zeros(1, pause_samples))
            continue

        emotion = None
        if role_override:
            role = role_override
            if line.startswith("("):
                match = re.match(r"\([^|)]*(?:\|([^)]+))?\)", line)
                if match:
                    emotion = match.group(1).strip() if match.group(1) else None
                    text = text[match.end() :].strip()
        else:
            role = None
            if line.startswith("("):
                match = re.match(r"\(([^)]+)\)", line)
                if match:
                    content = match.group(1)
                    if "|" in content:
                        role_part, emotion_part = content.split("|", 1)
                        role = role_part.strip()
                        emotion = emotion_part.strip()
                    else:
                        emotion = content.strip()
                    text = text[match.end() :].strip()

        if role_override:
            config = override_config
        else:
            if not role:
                first_role = next(iter(role_config.keys()), None)
                if not first_role:
                    raise ValueError(
                        f"第 {line_num + 1} 行缺少角色标识且配置文件中无可用角色"
                    )
                role = first_role
            if role not in role_config:
                raise ValueError(f"未定义的角色: {role} (第 {line_num + 1} 行)")
            config = role_config[role]

        emotion = emotion or config["default_emotion"]

        if method == "instruct":
            instruction = (
                f"用{emotion}的语气说这句话" if emotion else "用自然的语气说这句话"
            )
            generator = cosyvoice.inference_instruct2(
                text, instruction, config["prompt_speech"], stream=False
            )
        elif method == "zero_shot":
            generator = cosyvoice.inference_zero_shot(
                text,
                os.path.splitext(os.path.basename(config["prompt_path"]))[0],
                config["prompt_speech"],
                stream=False,
            )

        for i, j in enumerate(generator):
            audio_data = j["tts_speech"]
            if merge_files:
                all_audio.append(audio_data)

            if not merge_files:
                filename = get_unique_filename(
                    output_dir, base_name, "wav", line_num=line_num + 1
                )
                torchaudio.save(filename, audio_data, sample_rate)

    # 保存合并文件
    if merge_files and all_audio:
        combined = torch.cat(all_audio, dim=1)
        output_path = get_unique_filename(output_dir, base_name, "wav")
        torchaudio.save(output_path, combined, sample_rate)
        return output_path
    return None


def create_ui():
    """创建Gradio界面"""
    import json

    import gradio as gr

    # 加载角色配置以获取角色列表
    with open("role_config.json", "r", encoding="utf-8") as f:
        role_config = json.load(f)
    role_list = list(role_config.keys())

    def process_ui(text, role, method, merge_files):
        try:
            output_path = process_text_input(
                text=text,
                config_path="role_config.json",
                output_dir="output",
                merge_files=merge_files,
                role_override=role if role != "无" else None,
                method=method,
            )
            return output_path if output_path else "音频生成完成，请查看output目录"
        except Exception as e:
            return f"错误: {str(e)}"

    with gr.Blocks(title="CosyVoice语音合成") as demo:
        gr.Markdown("# CosyVoice语音合成系统")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="在此输入要合成的文本...\n支持多行文本\n可以使用(角色|情绪)格式指定角色和情绪",
                    lines=10,
                )
                role_dropdown = gr.Dropdown(
                    choices=["无"] + role_list,
                    value="无",
                    label="覆盖角色（选择后将忽略文本中的角色标记）",
                )
                method_radio = gr.Radio(
                    choices=["zero_shot", "instruct"],
                    value="zero_shot",
                    label="合成方法",
                )
                merge_checkbox = gr.Checkbox(value=True, label="合并音频文件")
                submit_btn = gr.Button("开始合成")

            with gr.Column():
                output_text = gr.Textbox(label="处理结果")
                audio_output = gr.Audio(label="合成音频")

        submit_btn.click(
            fn=process_ui,
            inputs=[text_input, role_dropdown, method_radio, merge_checkbox],
            outputs=[audio_output],
        )

    return demo


if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        # 使用原有的命令行处理逻辑
        parser = argparse.ArgumentParser(description="CosyVoice 语音合成工具")
        parser.add_argument("input", help="输入文本文件路径")
        parser.add_argument(
            "-c",
            "--config",
            default="role_config.json",
            help="角色配置文件路径（默认为role_config.json）",
        )
        parser.add_argument(
            "-o", "--output", default="outputs", help="输出目录（默认为output）"
        )
        parser.add_argument("-r", "--role", help="强制使用指定角色")
        parser.add_argument(
            "-m",
            "--method",
            choices=["instruct", "zero_shot"],
            default="zero_shot",
            help="合成方法（默认zero_shot）",
        )
        parser.add_argument(
            "--no-merge",
            action="store_false",  # 反转逻辑
            dest="merge",  # 映射到args.merge
            help="禁用合并功能（默认自动合并）",
        )
        args = parser.parse_args()
        process_txt_file(
            file_path=args.input,
            config_path=args.config,
            output_dir=args.output,
            merge_files=args.merge,
            role_override=args.role,
            method=args.method,
        )
    else:
        # 启动Gradio界面
        demo = create_ui()
        demo.launch()


# 基本用法
# python gen_cmd.py input.txt

# 高级用法
# python gen_cmd.py input.txt -c role_config.json \
#   -o results \
#   -r 晓辰 \
#   -m zero_shot \
#   --merge
