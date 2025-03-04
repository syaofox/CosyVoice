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


def process_txt_file(
    file_path,
    config_path,  # 改为配置文件路径
    merge_files=True,
):
    # 加载角色配置
    role_config = load_role_config(config_path)

    all_audio = []
    sample_rate = cosyvoice.sample_rate

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

        # 解析角色和情绪
        role, emotion = None, None
        if line.startswith("("):
            match = re.match(r"\(([^|)]+)(?:\|([^)]+))?\)", line)
            if match:
                role = match.group(1).strip()
                emotion = match.group(2).strip() if match.group(2) else None
                text = text[match.end() :].strip()

        # 获取角色配置
        if not role:
            raise ValueError(f"第 {line_num + 1} 行缺少角色标识")

        if role not in role_config:
            raise ValueError(f"未定义的角色: {role} (第 {line_num + 1} 行)")

        config = role_config[role]
        emotion = emotion or config["default_emotion"]

        # 执行合成
        if config["method"] == "instruct":
            instruction = (
                f"用{emotion}的语气说这句话" if emotion else "用自然的语气说这句话"
            )
            generator = cosyvoice.inference_instruct2(
                text, instruction, config["prompt_speech"], stream=False
            )
        elif config["method"] == "zero_shot":
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

    if merge_files and all_audio:
        combined = torch.cat(all_audio, dim=1)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"combined_{timestamp}.wav"
        torchaudio.save(output_path, combined, sample_rate)
        print(f"合并后的文件已保存至: {output_path}")


# 使用示例
process_txt_file(
    file_path=r"D:\downloads\spk2info\fanwen.txt",
    config_path="role_config.json",  # 改为配置文件路径
)
