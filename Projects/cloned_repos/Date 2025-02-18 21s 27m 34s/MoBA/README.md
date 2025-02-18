<p align="center">
  <a href="MoBA_Tech_Report.pdf"><img width="80%" src="figures/banner.png"></a>
</p>

# MoBA: Mixture of Block Attention for Long-Context LLMs

<p align="center">
  <a href="MoBA_Tech_Report.pdf"><img src="figures/logo.png" height="16" width="16" style="vertical-align:middle"><b> Full Report</b></a>
</p>

🚀 Introducing **MoBA --- Mixture of Block Attention**

* **Trainable Block Sparse Attention**: The full context is divided into blocks, where each query token learns to attend to the most relevant KV blocks, enabling efficient processing of long sequences.
* **Parameter-less Gating Mechanism**: A novel Parameter-less top-k gating mechanism is introduced to selects the most relevant blocks for each query token, ensuring that the model focuses only on the most informative blocks.
* **Seamlessly Transition between Full and Sparse Attention**: MoBA is designed to be a flexible substitute for full attention, allowing seamless transitions between full and sparse attention modes.
<p align="center">
  <img width="40%" src="figures/running_example.png" style="display:inline-block; margin-right:2%">
  <img width="40%" src="figures/moba_with_flash_attn.png" style="display:inline-block">
</p>


## Abstract
Scaling the effective context length is essential for advancing large language models (LLMs) toward artificial general intelligence (AGI). However, the quadratic increase in computational complexity inherent in traditional attention mechanisms presents a prohibitive overhead. Existing approaches either impose strongly biased structures, such as sink or window attention which are task-specific, or radically modify the attention mechanism into linear approximations, whose performance in complex reasoning tasks remains inadequately explored.
In this work, we propose a solution that adheres to the **“less structure”** principle, allowing the model to autonomously determine where to attend, rather than introducing predefined biases. We introduce Mixture of Block Attention (MoBA), an innovative approach that applies the principles of Mixture of Experts (MoE) to the attention mechanism. This novel architecture demonstrates superior performance on long-context tasks while offering a key advantage: the ability to seamlessly transition between full and sparse attention, enhancing efficiency without the risk of compromising performance. MoBA has already been deployed to support Kimi’s long-context requests and demonstrates significant advancements in efficient attention computation for LLMs. 
Our code is available at [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA).
<p align="center">
  <img width="40%" src="figures/computation_time.png" style="display:inline-block; margin-right:2%">
</p>

### Evaluation with 1M context length

<p align="center">
  <img width="80%" src="figures/needle-in-a-haystack.png">
</p>




## Environment Setup
**Note that current kernel implementations rely on `flash-attn==2.6.3`**

```bash
conda create -n moba python=3.10
conda activate moba
pip install .
```

## Quick Start
We provide a transformers-friendly implementation for MoBA.

Feel free to choose attention backends by `--attn` between `moba` and `moba_naive`.

```bash
python3 examples/llama.py --model meta-llama/Llama-3.1-8B --attn moba
```

## Unit Tests
```bash
pytest tests/test_moba_attn.py
```

## References
* Llama Implementation: [huggingface/transformers](https://github.com/huggingface/transformers)
* Flash Attention: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)



## Citation
If you find MoBA is useful or want to use in your projects, please kindly cite our paper:
```
@article{MoonshotMoBA,
  author = {Lu, Enzhe and Jiang, Zhejun and Liu, Jingyuan and Du, Yulun and Jiang, Tao and Hong, Chao and Liu, Shaowei and He, Weiran and Yuan, Enming and Wang, Yuzhi and Huang, Zhiqi and Yuan, Huan and Xu, Suting and Xu, Xinran and Lai, Guokun and Chen, Yanru and Zheng, Huabin and Yan, Junjie and Su, Jianlin and Wu, Yuxin and Zhang, Neo Y. and Yang, Zhilin and Zhou, Xinyu and Zhang, Mingxing and Qiu, Jiezhong},
  title = {MoBA: Mixture of Block Attention for Long-Context LLMs},
  year = {2025},
}
```
