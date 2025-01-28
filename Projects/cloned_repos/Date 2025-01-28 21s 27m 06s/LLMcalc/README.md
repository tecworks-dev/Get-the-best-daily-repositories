# LLM Calculator

This script estimates the memory requirements and performance of Hugging Face models based on quantization levels. It fetches model parameters, calculates required memory, and analyzes performance with different RAM/VRAM configurations.

It supports windows and Linux, AMD, Intel, and Nvidia GPUs. You will need smi (cuda toolkit (?)) installed to recognise your Nvidia GPU.

Warning: The tool isn't tested outside of Linux+Nvidia, so results may be inaccurate. It's a rough estimate.
It may or may not work with MultiGPU setups. If not, use -n to specify how many cards you have (assuming they're all the same). If you have an RTX 3070 and an RTX 3060 for example, you may need to specify -v and -b to be the average values of the two. 

**Flags**
```
-b, --bandwidth: Override memory bandwidth (GB/s).
-n, --num-gpus: Number of GPUs (default is 1).
-v, --vram: Override VRAM amount per card in GB.
```
Example:
`python main.py -b 950 -n 2 -v 24`


## Dependencies
You'll need these to run it in python. 3.12.3 is what I wrote it in, but any modern version should work.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run with:

`uv run main.py`

For AMD + Linux you'll need `sudo apt install pciutils`

Nvidia will need drivers, as long as `nvidia-smi` works this program should.

Intel needs `lspci`, dunno if that supports windows.



## How It Works
Enter a Hugging Face model ID (e.g., microsoft/phi-4) to get its parameter count.
The script fetches system RAM and VRAM specs. You can override them with flags.
It analyzes memory requirements for several quantization schemes and estimates throughput (tk/s).

![GUI Preview](gui.png)

Hover over a cell to see how many layers you need to offload, and what context you can fit without KV cache quantizing.


**Cli Output**

```
Enter Hugging Face model ID (e.g., microsoft/phi-4): microsoft/phi-4
Model Parameters: 14.7B params (14.70B params)
Total RAM: 33.53 GB
VRAM: 8.00 GB, ~448.0GB/s
Estimated RAM Bandwidth: 64.00 GB/s

Analysis for each quantization level:

FP8:
Run Type: Partial offload
Memory Required: 16.43 GB
GPU Offload Percentage: 48.7%
Estimated tk/s: 5.38

Q6_K_S:
Run Type: Partial offload
Memory Required: 13.86 GB
GPU Offload Percentage: 57.7%
Estimated tk/s: 7.39

Q5_K_S:
Run Type: Partial offload
Memory Required: 11.84 GB
GPU Offload Percentage: 67.6%
Estimated tk/s: 10.63

Q4_K_M:
Run Type: Partial offload
Memory Required: 10.55 GB
GPU Offload Percentage: 75.8%
Estimated tk/s: 14.71

IQ4_XS:
Run Type: Partial offload
Memory Required: 9.64 GB
GPU Offload Percentage: 83.0%
Estimated tk/s: 19.92

Q3_K_M:
Run Type: KV cache offload
Memory Required: 8.90 GB
Estimated tk/s: 45.30

IQ3_XS:
Run Type: All in VRAM
Memory Required: 7.80 GB
Estimated tk/s: 57.45

IQ2_XS:
Run Type: All in VRAM
Memory Required: 6.14 GB
Estimated tk/s: 72.90
```
