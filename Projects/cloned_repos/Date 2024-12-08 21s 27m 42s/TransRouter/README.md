# TransRouter

TransRouter 是一个实时语音翻译工具，支持中英文双向翻译。它使用 Azure Speech Services 进行语音识别，OpenAI GPT-4 进行翻译，可以直接与 Zoom 等会议软件集成。

## 功能特点

- 实时语音识别和翻译
- 中英文双向翻译
- 自动语音合成
- 支持一键切换识别语言
- 与 Zoom 等会议软件无缝集成
- 低延迟的流式处理
- 自动保存原始录音和合成音频

## 系统要求

- Python 3.8 或更高版本
- 只支持Mac
- BlackHole 虚拟音频设备（用于音频路由）
- 稳定的网络连接
- Azure Speech Services 账号
- OpenAI API 密钥

## 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/TransRouter.git
cd TransRouter
```


2. 创建并激活虚拟环境：

Mac:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```



4. 配置环境变量：
   - 复制 `.env.example` 为 `.env`
   - 填入您的 API 密钥：

```bash
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=your_openai_api_base # 可选，如果使用自定义 API 端点
```


## 音频设备配置

### macOS
1. 安装 BlackHole：

```bash
brew install blackhole-2ch
```

2. 系统设置：
   - 打开系统偏好设置 > 声音
   - 确认可以看到 BlackHole 2ch 设备



### Zoom 配置
1. 打开 Zoom 设置 > 音频
2. 麦克风：选择系统默认麦克风
3. 扬声器：选择 "BlackHole 2ch"


## 使用说明

1. 启动程序：

```bash
python transrouter.py
```

2. 程序功能：
   - 默认模式：识别中文并翻译为英文  
   - 按 Ctrl+C：停止程序

3. 音频文件：
   - 原始录音保存在 `recordings` 目录
   - 合成语音保存在 `synthesis` 目录
   - 文件格式：16kHz 采样率，16bit 深度，单声道 WAV
  

## 常见问题

1. 找不到音频设备：
   - 检查 BlackHole 是否正确安装
   - 运行程序时查看打印的设备列表
   - 确认系统音频设置中可以看到虚拟设备

2. 识别不准确：
   - 确保使用正确的语言模式
   - 检查麦克风音量和环境噪音
   - 说话时保持适当距离和语速

3. 翻译延迟：
   - 检查网络连接
   - 可能是 API 调用限制
   - 尝试调整 VAD 超时设置

4. 音频问题：
   - 确认采样率设置（16kHz）
   - 检查音频设备路由
   - 验证 Zoom 音频设置

## 开发说明

- 语音识别：使用 Azure Speech Services 的流式识别
- 文本翻译：使用 OpenAI GPT-4 模型
- 语音合成：使用 Azure Speech Services 的神经网络语音
- 音频处理：使用 sounddevice 和 numpy 处理音频流

## 注意事项

1. API 使用：
   - 注意 API 调用限制和计费
   - 保护好 API 密钥
   - 建议使用 API 代理

2. 音频设置：
   - 使用 16kHz 采样率
   - 单声道音频
   - PCM 16bit 格式

3. 系统要求：
   - 确保 Python 环境正确
   - 安装必要的音频驱动
   - 保持充足的系统资源
