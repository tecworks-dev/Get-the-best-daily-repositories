<h1 align='center'>HelloMeme: Integrating Spatial Knitting Attentions to Embed High-Level and Fidelity-Rich Conditions in Diffusion Models</h1>

<div align='center'>
    <a href='https://github.com/songkey' target='_blank'>Shengkai Zhang</a>, <a href='https://github.com/RhythmJnh' target='_blank'>Nianhong Jiao</a>, <a href='https://github.com/Shelton0215' target='_blank'>Tian Li</a>, <a href='https://github.com/chaojie12131243' target='_blank'>Chaojie Yang</a>, <a href='https://github.com/xchgit' target='_blank'>Chenhui Xue</a><sup>*</sup>, <a href='https://github.com/boya34' target='_blank'>Boya Niu</a><sup>*</sup>, <a href='https://github.com/HelloVision/HelloMeme' target='_blank'>Jun Gao</a> 
</div>

<div align='center'>
    HelloVision | HelloGroup Inc.
</div>

<div align='center'>
    <small><sup>*</sup> Intern</small>
</div>

<br>
<div align='center'>
    <a href='https://songkey.github.io/hellomeme/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
    <a href='https://arxiv.org/pdf/2410.22901'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/songkey'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='https://github.com/HelloVision/HelloMeme'><img src='https://img.shields.io/badge/GitHub-Code-blue'></a>
</div>


## 🔆 New Features/Updates

- ✅ `11/6/2024` The face proportion in the reference image significantly affects the generation quality. We have encapsulated the **recommended image cropping method** used during training into a `CropReferenceImage` Node. Refer to the workflows in the `ComfyUI_HelloMeme/workflows directory`: `hellomeme_video_cropref_workflow.json` and `hellomeme_image_cropref_workflow.json`.


## Introduction

This repository is the official implementation of the [`HelloMeme`](https://arxiv.org/pdf/2410.22901) ComfyUI interface, featuring both image and video generation functionalities. Example workflow files can be found in the `ComfyUI_HelloMeme/workflows` directory. Test images and videos are saved in the `ComfyUI_HelloMeme/examples` directory. Below are screenshots of the interfaces for image and video generation.

### Image Generation Interface

<p align="center">
  <img src="workflows/hellomeme_image_example.jpg" alt="image_generation_interface">
</p>

### Video Generation Interface

<p align="center">
  <img src="workflows/hellomeme_video_example.jpg" alt="video_generation_interface">
</p>

