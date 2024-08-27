import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.append("{}/third_party/AcademiCodec".format(ROOT_DIR))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio


cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
prompt_speech_16k = load_wav(r"D:\aisound\音色\晓辰\雄大的反应十分平静，声称那些只不过是谣言。.wav", 16000)
output = cosyvoice.inference_zero_shot('咱们这位海军上尉阿帅，听起来就像是个从海军肥皂剧中走出来的角色，25年的海上航船经验，让他觉得这件事儿绝不像是看起来那么单纯。', '雄大的反应十分平静，声称那些只不过是谣言。', prompt_speech_16k)
torchaudio.save(r'D:\aisound\temp\晓辰1.wav', output['tts_speech'], 22050) # type: ignore

output = cosyvoice.inference_zero_shot("他想要去那艘神秘的船上探探究竟，结果遭到了阿伟的连环劝退，阿伟同学，你是怕他揭穿你的小把戏吗？最终，阿帅同志只能暂时放下他的侦探梦，让阿伟去休息。", '雄大的反应十分平静，声称那些只不过是谣言。', prompt_speech_16k)
torchaudio.save(r'D:\aisound\temp\晓辰2.wav', output['tts_speech'], 22050) # type: ignore

