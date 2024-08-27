import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.append("{}/third_party/AcademiCodec".format(ROOT_DIR))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch

import gradio as gr

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
prompt_speech_16k = load_wav(r"D:\aisound\音色\林志玲\他会看看你在这个业界的成绩，然后知道你真的在做什么。.wav", 16000)
prompt_text = "他会看看你在这个业界的成绩，然后知道你真的在做什么。"

article = """
阿金正在家忙活着打扫，门外忽然响起一阵砸门声，震得屋子都晃了几晃。
门一开，一个陌生的壮汉站在门口，态度嚣张地命令他下去挪车，说是占了他的专属车位。
阿金虽心里不爽，但还是下楼去了，刚坐上车，一滩新鲜的口水赫然出现在车窗上，显然是那壮汉的杰作。
阿金怒了，正要发作，却瞥见对方粗壮的胳膊上纹着猛兽纹身，立马就没了脾气。
"""

# 初始化一个空的音频张量
merged_audio = torch.tensor([])

# 假设采样率为22050
sample_rate = 22050

texts = article.splitlines()
for text in texts:
    if not text:
        continue

    output = cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)
    audio = output['tts_speech']
    
    # 确保音频数据的维度正确
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # 合并音频
    if merged_audio.numel() == 0:
        merged_audio = audio
    else:
        merged_audio = torch.cat((merged_audio, audio), dim=1)

# 保存合并后的音频
output_path = r'D:\aisound\temp\林志玲_merged.wav'
torchaudio.save(output_path, merged_audio, sample_rate) # type: ignore

