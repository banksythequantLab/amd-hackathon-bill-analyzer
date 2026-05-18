@echo off
REM Pull qwen3:30b-a3b-instruct-2507-q4_K_M as a background subprocess.
REM Stdout/stderr go to eval\ollama-pull-instruct.log so we can tail it.
cd /d B:\amd-hackathon-bill-analyzer-3090
"C:\Users\solti\AppData\Local\Programs\Ollama\ollama.exe" pull qwen3:30b-a3b-instruct-2507-q4_K_M > eval\ollama-pull-instruct.log 2>&1
echo DONE >> eval\ollama-pull-instruct.log
