import argparse
import datetime
import os
import re
import shutil
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
        """解析单行文本，支持多种格式
        支持格式：
        1. 纯文本 - 使用默认角色和默认情绪
        2. (角色)文本 - 指定角色，使用默认情绪
        3. (角色|情绪)文本 - 指定角色和情绪
        4. (|情绪)文本 - 使用默认角色，指定情绪

        返回: (角色, 情绪, 文本内容)
        """
        text = line.strip()
        role = emotion = None

        # 检查是否有括号标记
        if text.startswith("("):
            # 使用非贪婪匹配来处理可能的嵌套括号
            match = re.match(r"\((.*?)\)(.*)", text)
            if match:
                content = match.group(1).strip()
                text = match.group(2).strip()

                # 解析括号内容
                if "|" in content:
                    # 格式: (角色|情绪) 或 (|情绪)
                    parts = [p.strip() for p in content.split("|", 1)]
                    role_part = parts[0]
                    emotion = parts[1]

                    # 处理可能的空值
                    if role_part:
                        role = role_part
                else:
                    # 格式: (角色)
                    role = content

        # 清理文本中的方括号表情标记
        text = re.sub(r"\[.*?\]", "", text).strip()

        # 确保所有返回值都经过清理
        if role:
            role = role.strip()
        if emotion:
            emotion = emotion.strip()

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
        self.default_role = None  # 添加默认角色属性

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
            if not self.default_role:
                raise ValueError(f"第 {line_num + 1} 行缺少角色标识，且未设置默认角色")
            role = self.default_role
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
        output_dir: str = "outputs",
        merge_files: bool = True,
        role_override: Optional[str] = None,
        method: str = "instruct",
        keep_temp: bool = False,
        default_role: Optional[str] = None,  # 添加默认角色参数
    ) -> Optional[str]:
        """处理文本并生成音频"""
        self.default_role = default_role  # 设置默认角色
        if role_override and role_override not in self.role_config:
            raise ValueError(f"覆盖角色不存在于配置中: {role_override}")
        if not default_role:
            raise ValueError("必须指定默认角色")

        os.makedirs(output_dir, exist_ok=True)
        lines = text.strip().split("\n")
        first_line = next((l for l in lines if l.strip()), "").strip()
        base_name = TextProcessor.sanitize_filename(
            re.sub(r"^\([^)]*\)", "", first_line)
        )

        all_audio = []
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        try:
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
                    role, config = self.get_role_config(role, line_num)

                emotion = emotion or config["default_emotion"]
                audio_data = self.generate_audio(content, emotion, config, method)

                # 始终先保存独立文件（即使合并模式）
                temp_path = os.path.join(temp_dir, f"line_{line_num + 1}.wav")
                torchaudio.save(temp_path, audio_data, self.sample_rate)

                if merge_files:
                    all_audio.append(audio_data)

            # 合并成功后清理临时文件
            if merge_files and all_audio and not keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            # 异常时保留已生成的临时文件
            error_info = (
                f"处理中断于第 {line_num + 1} 行\n"
                f"错误原因: {str(e)}\n"
                f"已生成段落保存至: {temp_dir}"
            )
            if merge_files and all_audio:
                partial_output = self._save_partial_audio(all_audio, output_dir)
                error_info += f"\n部分合并文件: {partial_output}"
            raise RuntimeError(error_info) from e

        if merge_files and all_audio:
            combined = torch.cat(all_audio, dim=1)
            output_path = self.get_unique_filename(output_dir, base_name, "wav")
            torchaudio.save(output_path, combined, self.sample_rate)
            return output_path
        return None

    def _save_partial_audio(self, audio_chunks, output_dir):
        """保存已处理的部分音频"""
        combined = torch.cat(audio_chunks, dim=1)
        output_path = os.path.join(output_dir, "PARTIAL_OUTPUT.wav")
        torchaudio.save(output_path, combined, self.sample_rate)
        return output_path

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
            text: str,
            role: str,
            default_role: str,  # 新增默认角色参数
            method: str,
            merge_files: bool,
            keep_temp: bool,
        ) -> Tuple[str, Optional[str]]:
            try:
                output_path = self.tts_processor.process_text(
                    text=text,
                    output_dir="outputs",
                    merge_files=merge_files,
                    role_override=role if role != "无" else None,
                    method=method,
                    keep_temp=keep_temp,
                    default_role=default_role,  # 传递默认角色
                )
                message = "✅ 音频生成完成" + (
                    "（已保留临时文件）" if keep_temp else ""
                )
                return message, output_path
            except Exception as e:
                error_msg = f"❌ 错误: {str(e)}\n临时文件保存在: outputs/temp"
                return error_msg, None

        with gr.Blocks(title="CosyVoice语音合成") as demo:
            gr.Markdown("# CosyVoice语音合成系统")

            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="输入文本",
                        placeholder="在此输入要合成的文本...\n支持多行文本\n支持格式：\n1. 纯文本 - 使用默认角色和默认情绪\n2. (角色)文本 - 指定角色，使用默认情绪\n3. (角色|情绪)文本 - 指定角色和情绪\n4. (|情绪)文本 - 使用默认角色，指定情绪",
                        lines=10,
                    )

                    # 添加默认角色下拉框
                    default_role_dropdown = gr.Dropdown(
                        choices=list(self.tts_processor.role_config.keys()),
                        value=list(self.tts_processor.role_config.keys())[0],
                        label="默认角色（当文本未指定角色时使用）",
                        interactive=True,
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
                    keep_temp_check = gr.Checkbox(
                        label="保留所有临时文件", info="即使成功完成也保留临时文件"
                    )
                    submit_btn = gr.Button("开始合成", variant="primary")

                with gr.Column():
                    output_text = gr.Textbox(label="处理结果")
                    audio_output = gr.Audio(label="合成音频")

            submit_btn.click(
                fn=process_ui,
                inputs=[
                    text_input,
                    role_dropdown,
                    default_role_dropdown,  # 添加默认角色输入
                    method_radio,
                    merge_checkbox,
                    keep_temp_check,
                ],
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
        parser.add_argument("-o", "--output", default="outputs", help="输出目录")
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
        parser.add_argument(
            "--keep-temp", action="store_true", help="保留临时文件（默认自动清理）"
        )
        parser.add_argument(
            "-d",
            "--default-role",
            help="设置默认角色（当文本未指定角色时使用）",
            required=True,
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
            keep_temp=args.keep_temp,
            default_role=args.default_role,  # 添加默认角色参数
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
# # 处理中断时
# 错误: 处理中断于第 23 行
# 错误原因: 未定义的角色: 晓晨
# 已生成段落保存至: outputs/temp
# 部分合并文件: outputs/PARTIAL_OUTPUT.wav

# # 恢复建议
# 1. 修复第23行错误
# 2. 重新运行时添加：--start-from 23 --temp-dir outputs/temp
