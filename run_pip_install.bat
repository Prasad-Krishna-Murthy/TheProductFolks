@echo off
"D:\Artificial Intelligence\ChatGPT\TheProductFolks\venv\Scripts\python.exe" -m pip install transformers peft accelerate sentencepiece gradio langdetect pandas numpy scikit-learn sacrebleu rapidfuzz > "D:\Artificial Intelligence\ChatGPT\TheProductFolks\pip_task_out.txt" 2>&1
echo DONE_MARKER >> "D:\Artificial Intelligence\ChatGPT\TheProductFolks\pip_task_out.txt"
