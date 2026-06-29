sudo sysctl iogpu.wired_limit_mb=184320
./ds4-server --ctx 100000  -m /Users/yuhai/github/ds4/gguf/DeepSeek-V4-Flash-4Expert-Q4K.gguf --port 8000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
