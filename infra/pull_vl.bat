@echo off
REM 3090 FORK: Pull qwen3-vl:8b-instruct from Ollama.
REM Stdout/stderr go to eval\ollama-pull-vl.log; "DONE" appended on success.
cd /d B:\amd-hackathon-bill-analyzer-3090
"C:\Users\solti\AppData\Local\Programs\Ollama\ollama.exe" pull qwen3-vl:8b-instruct > eval\ollama-pull-vl.log 2>&1
echo DONE >> eval\ollama-pull-vl.log
