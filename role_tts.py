import argparse
import datetime
import os
import re
import sys
from typing import List, Optional, Tuple

sys.path.append("third_party/Matcha-TTS")
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 初始化TTS模型
cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B", load_jit=False, load_trt=False, fp16=False
)


class TextProcessor:
    """文本处理类"""

    @staticmethod
    def parse_line(line: str) -> Tuple[Optional[str], Optional[str], str]:
        """解析单行文本，返回(角色, 情绪, 文本内容)"""
        text = line.strip()
        role = emotion = None

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

        return role, emotion, text

    @staticmethod
    def sanitize_filename(text: str) -> str:
        """生成合法文件名"""
        clean_text = re.sub(r'[\\/*?:"<>|()]', "", text)
        clean_text = clean_text[:10].strip()
        if not clean_text:
            return datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return clean_text


class TTSProcessor:
    """TTS处理类"""

    def __init__(self, config_path: str):
        self.role_config = self.load_role_config(config_path)
        self.sample_rate = cosyvoice.sample_rate

    @staticmethod
    def load_role_config(config_path: str) -> dict:
        """加载角色配置"""
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        for role in config.values():
            try:
                role["prompt_speech"] = load_wav(role["prompt_path"], 16000)
            except Exception as e:
                raise ValueError(f"加载角色音频失败 [{role['prompt_path']}]: {str(e)}")
        return config

    def get_role_config(self, role: str, line_num: int) -> Tuple[str, dict]:
        """获取角色配置"""
        if not role:
            first_role = next(iter(self.role_config.keys()), None)
            if not first_role:
                raise ValueError(
                    f"第 {line_num + 1} 行缺少角色标识且配置文件中无可用角色"
                )
            role = first_role
        if role not in self.role_config:
            raise ValueError(f"未定义的角色: {role} (第 {line_num + 1} 行)")
        return role, self.role_config[role]

    def generate_audio(
        self, text: str, emotion: str, config: dict, method: str
    ) -> torch.Tensor:
        """生成音频"""
        if method == "instruct":
            instruction = (
                f"用{emotion}的语气说这句话" if emotion else "用自然的语气说这句话"
            )
            generator = cosyvoice.inference_instruct2(
                text, instruction, config["prompt_speech"], stream=False
            )
        else:  # zero_shot
            generator = cosyvoice.inference_zero_shot(
                text,
                os.path.splitext(os.path.basename(config["prompt_path"]))[0],
                config["prompt_speech"],
                stream=False,
            )

        return next(generator)["tts_speech"]

    def process_text(
        self,
        text: str,
        output_dir: str = "output",
        merge_files: bool = True,
        role_override: Optional[str] = None,
        method: str = "instruct",
    ) -> Optional[str]:
        """处理文本并生成音频"""
        if role_override and role_override not in self.role_config:
            raise ValueError(f"覆盖角色不存在于配置中: {role_override}")

        os.makedirs(output_dir, exist_ok=True)
        lines = text.strip().split("\n")
        first_line = next((l for l in lines if l.strip()), "").strip()
        base_name = TextProcessor.sanitize_filename(
            re.sub(r"^\([^)]*\)", "", first_line)
        )

        all_audio = []
        for line_num, line in enumerate(lines):
            text = line.strip()
            if not text:
                if merge_files:
                    all_audio.append(torch.zeros(1, int(0.8 * self.sample_rate)))
                continue

            role, emotion, content = TextProcessor.parse_line(line)
            if role_override:
                role = role_override
                config = self.role_config[role_override]
            else:
                assert role, "找不到有效角色"
                role, config = self.get_role_config(role, line_num)

            emotion = emotion or config["default_emotion"]
            audio_data = self.generate_audio(content, emotion, config, method)

            if merge_files:
                all_audio.append(audio_data)
            else:
                filename = self.get_unique_filename(
                    output_dir, base_name, "wav", line_num + 1
                )
                torchaudio.save(filename, audio_data, self.sample_rate)

        if merge_files and all_audio:
            combined = torch.cat(all_audio, dim=1)
            output_path = self.get_unique_filename(output_dir, base_name, "wav")
            torchaudio.save(output_path, combined, self.sample_rate)
            return output_path
        return None

    @staticmethod
    def get_unique_filename(
        directory: str, base_name: str, extension: str, line_num: Optional[int] = None
    ) -> str:
        """生成不重复的文件名"""
        counter = 1
        while True:
            main_part = (
                f"{base_name}_{line_num}"
                if line_num is not None
                else f"{base_name}_combined"
            )
            full_name = (
                f"{main_part}_{counter}.{extension}"
                if counter > 1
                else f"{main_part}.{extension}"
            )
            path = os.path.join(directory, full_name)
            if not os.path.exists(path):
                return path
            counter += 1


class UI:
    """UI界面类"""

    def __init__(self, tts_processor: TTSProcessor):
        self.tts_processor = tts_processor

    def create(self):
        """创建Gradio界面"""
        import gradio as gr

        def process_ui(
            text: str, role: str, method: str, merge_files: bool
        ) -> Tuple[str, Optional[str]]:
            try:
                output_path = self.tts_processor.process_text(
                    text=text,
                    output_dir="output",
                    merge_files=merge_files,
                    role_override=role if role != "无" else None,
                    method=method,
                )
                message = (
                    "音频生成完成，请查看output目录"
                    if not output_path
                    else "音频生成成功"
                )
                return message, output_path  # 返回两个值：状态消息和音频路径
            except Exception as e:
                error_message = f"错误: {str(e)}"
                return error_message, None  # 发生错误时返回错误消息和None

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
                        choices=["无"] + list(self.tts_processor.role_config.keys()),
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
                outputs=[output_text, audio_output],
            )

        return demo


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 命令行模式
        parser = argparse.ArgumentParser(description="CosyVoice 语音合成工具")
        parser.add_argument("input", help="输入文本文件路径")
        parser.add_argument(
            "-c", "--config", default="role_config.json", help="角色配置文件路径"
        )
        parser.add_argument("-o", "--output", default="output", help="输出目录")
        parser.add_argument("-r", "--role", help="强制使用指定角色")
        parser.add_argument(
            "-m",
            "--method",
            choices=["instruct", "zero_shot"],
            default="zero_shot",
            help="合成方法",
        )
        parser.add_argument(
            "--no-merge",
            action="store_false",
            dest="merge",
            help="禁用合并功能",
        )
        args = parser.parse_args()

        # 读取输入文件
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()

        # 处理文本
        tts_processor = TTSProcessor(args.config)
        tts_processor.process_text(
            text=text,
            output_dir=args.output,
            merge_files=args.merge,
            role_override=args.role,
            method=args.method,
        )
    else:
        # UI模式
        tts_processor = TTSProcessor("role_config.json")
        ui = UI(tts_processor)
        demo = ui.create()
        demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()


# 基本用法
# python gen_cmd.py input.txt

# 高级用法
# python gen_cmd.py input.txt -c role_config.json \
#   -o results \
#   -r 晓辰 \
#   -m zero_shot \
#   --merge
