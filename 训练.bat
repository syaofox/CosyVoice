@echo off
chcp 65001
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\hf_download
call conda activate cosyvoice
echo "欢迎使用由B站CyberWon开发的CosyVoice预训练脚本。"
echo "1.使用方法：新增一个你想要的音色名文件夹，里面放一个单声道的wav格式的音频文件，音频文件名是参考音频对应的文本"
echo "2.音色名就是新加文件夹名称。"
echo "3.训练完成后，启动非Instruct模式，看添加成功了么。"
echo "勿用作商业用途，仅供学习研究使用。"
set /p AudioDir= 输入音频文件夹，目录下仅放一个音频文件: 
python train.py --audio_dir %AudioDir%

pause 
