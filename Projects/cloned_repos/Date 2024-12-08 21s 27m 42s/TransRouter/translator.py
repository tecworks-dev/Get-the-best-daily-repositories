import sounddevice as sd
import numpy as np
import threading
import queue
import asyncio
from speech_translator import SpeechTranslator
import keyboard
from scipy.io import wavfile
import os
from datetime import datetime


class AudioTranslator:
    def __init__(self, source_lang="zh-CN"):
        # 音频参数设置
        self.sample_rate = 16000  # 修改为16kHz，更适合语音识别
        self.channels = 1  # 单声道
        self.dtype = np.int16
        self.chunk_size = 1600  # 调整块大小为100ms的数据量(16000 * 0.1)
        self.buffer = queue.Queue()

        # 设备设置
        self.input_device = None  # 使用系统默认麦克风
        self.output_device = "BlackHole 2ch"  # 输出到 Zoom

        # 检查设备是否支持指定采样率
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        print(f"\n默认输入设备: {default_input['name']}")
        print(f"支持的采样率: {default_input['default_samplerate']}")

        # 初始化翻译器
        self.translator = SpeechTranslator(source_lang)

        # 打印可用设备信息，方便调试
        print("\n可用音频设备:")
        print(sd.query_devices())

        # 创建事件循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # 创建录音和合成音频的保存目录
        self.recordings_dir = "recordings"
        self.synthesis_dir = "synthesis"
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.synthesis_dir, exist_ok=True)

        # 用于保存音频数据的列表
        self.recording_buffer = []

    def save_wav(self, audio_data, directory, prefix=""):
        """保存音频数据为WAV文件"""
        if audio_data is None or len(audio_data) == 0:
            print("没有音频数据")
            return None

        # 生成文件名（使用时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(directory, f"{prefix}_{timestamp}.wav")

        # 保存为WAV文件
        wavfile.write(filename, self.sample_rate, audio_data)
        print(f"\n音频已保存: {filename}")
        return filename

    def save_audio(self):
        """保存录音到WAV文件"""
        if not self.recording_buffer:
            print("没有录音数据")
            return

        # 将所有音频数据合并为一个数组
        audio_data = np.concatenate(self.recording_buffer)
        self.save_wav(audio_data, self.recordings_dir, "recording")

        # 清空缓冲区
        self.recording_buffer = []

    def audio_callback(self, indata, frames, time, status):
        """音频回调函数，处理输入的音频数据"""
        if status:
            print(f"状态: {status}")
        # 将音频数据添加到录音缓冲区
        self.recording_buffer.append(indata.copy())
        self.buffer.put(indata.copy())

    async def process_audio(self, audio_data):
        """处理音频数据"""
        try:
            # 进行流式语音识别
            text = self.translator.transcribe_audio(audio_data)
            if text:
                # 翻译文本并合成语音
                translated_text, synthesized_audio = await self.translator.translate_text(text)
                print(f"原文: {text}")
                print(f"译文: {translated_text}")

                if synthesized_audio is not None:
                    # 保存合成的音频
                    self.save_wav(synthesized_audio,
                                  self.synthesis_dir, "synthesis")

                    # 播放合成的语音
                    sd.play(synthesized_audio, self.sample_rate,
                            device=self.output_device)
                    sd.wait()

        except Exception as e:
            print(f"处理音频时出错: {e}")

    def switch_language(self):
        """切换识别和翻译的语言"""
        self.translator.switch_language()

    def start_streaming(self):
        """开始音频流处理"""
        try:
            with sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                dtype=self.dtype,
                blocksize=self.chunk_size
            ):
                print("开始录音...（按 Ctrl+C 停止，按 Tab 键切换语言，按 S 键保存录音）")
                while True:
                    audio_data = self.buffer.get()
                    self.loop.run_until_complete(
                        self.process_audio(audio_data))

        except KeyboardInterrupt:
            print("\n停止录音")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            # 保存最后的录音
            self.save_audio()
            # 确保清理资源
            self.loop.close()
            if hasattr(self, 'translator'):
                del self.translator

    def run(self):
        """运行翻译器"""
        streaming_thread = threading.Thread(target=self.start_streaming)
        streaming_thread.start()
        return streaming_thread  # 返回线程对象以便控制


if __name__ == "__main__":
    translator = AudioTranslator()
    thread = translator.run()

    # 注册快捷键
    # keyboard.on_press_key("tab", lambda _: translator.switch_language())
    # keyboard.on_press_key("s", lambda _: translator.save_audio())

    try:
        thread.join()
    except KeyboardInterrupt:
        translator.save_audio()
        print("\n程序已停止")
