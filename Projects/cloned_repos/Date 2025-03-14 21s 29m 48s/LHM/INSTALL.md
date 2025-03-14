# Installation

## Requirements

- Linux
- Python 3.10
- Pytorch 2.3.0
- torchvision 0.18.0

## 1. Install pytorch
  ```bash
  # cuda 11.8
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
  # cuda 12.1
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
  ```

## 2. Install xformers
  ```bash
  # cuda 11.8
  pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118

  # cuda 12.1
  pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121
  ```

## 3. Install base dependencies
  ```bash
  pip install -r requirements.txt
  pip uninstall basicsr
  ```

## 4. Install SAM2 lib. We use the modified version.
```bash
pip install git+https://github.com/hitsz-zuoqi/sam2/

# or
cd ..
git clone --recursive https://github.com/hitsz-zuoqi/sam2
pip install ./sam2
```

## 5. Install 3DGS python-bings
```bash
cd ..

# we use the version modified by Jiaxiang Tang, thanks for this great job!
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
git clone https://github.com/camenduru/simple-knn.git
pip install ./simple-knn
```

## 6. Please then follow the [Pytorch3D](https://github.com/facebookresearch/pytorch3d) to install Pytorch3D lib.