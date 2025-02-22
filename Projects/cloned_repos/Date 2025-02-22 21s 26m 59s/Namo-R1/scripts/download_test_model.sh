mkdir checkpoints
cd checkpoints/
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir Qwen2.5-0.5B-Instruct
huggingface-cli download lucasjin/aimv2-large-patch14-224 --local-dir aimv2-large-patch14-224
huggingface-cli download lucasjin/aimv2-large-patch14-native --local-dir aimv2-large-patch14-native
