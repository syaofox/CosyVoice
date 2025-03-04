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
            # 仍允许解析情绪
            if line.startswith("("):
                match = re.match(
                    r"\([^|)]*(?:\|([^)]+))?\)", line
                )  # 修改正则只匹配情绪
                if match:
                    emotion = match.group(1).strip() if match.group(1) else None
                    text = text[match.end() :].strip()
        else:  # 原有解析逻辑
            role = None
            if line.startswith("("):
                match = re.match(r"\(([^|)]+)(?:\|([^)]+))?\)", line)
                if match:
                    role = match.group(1).strip()
                    emotion = match.group(2).strip() if match.group(2) else None
                    text = text[match.end() :].strip()

        # 获取角色配置（修改后的逻辑）
        if role_override:
            config = override_config
        else:
            if not role:
                raise ValueError(f"第 {line_num + 1} 行缺少角色标识")
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
                filename = f"{base_name}_{line_num + 1}.wav"
                save_path = os.path.join(output_dir, filename)
                torchaudio.save(save_path, audio_data, sample_rate)
                print(f"生成文件: {save_path}")

    # 保存合并文件逻辑
    if merge_files and all_audio:
        combined = torch.cat(all_audio, dim=1)
        filename = f"{base_name}_combined.wav"
        output_path = os.path.join(output_dir, filename)
        torchaudio.save(output_path, combined, sample_rate)
        print(f"合并文件已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice 语音合成工具")
    parser.add_argument("input", help="输入文本文件路径")
    parser.add_argument("-c", "--config", required=True, help="角色配置文件路径")
    parser.add_argument("-o", "--output", default="output", help="输出目录（默认为output）")
    parser.add_argument("-r", "--role", help="强制使用指定角色")
    parser.add_argument(
        "-m",
        "--method",
        choices=["instruct", "zero_shot"],
        default="instruct",
        help="合成方法（默认instruct）",
    )
    parser.add_argument("--merge", action="store_true", help="合并所有段落为单个文件")

    args = parser.parse_args()

    process_txt_file(
        file_path=args.input,
        config_path=args.config,
        output_dir=args.output,
        merge_files=args.merge,
        role_override=args.role,
        method=args.method,
    )


# 基本用法
# python gen_cmd.py input.txt -c role_config.json -o outputs

# 高级用法
# python gen_cmd.py input.txt -c role_config.json \
#   -o results \
#   -r 晓辰 \
#   -m zero_shot \
#   --merge