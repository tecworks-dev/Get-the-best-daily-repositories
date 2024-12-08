import azure.cognitiveservices.speech as speechsdk
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import queue
import asyncio
import numpy as np


class SpeechTranslator:
    def __init__(self, source_lang="zh-CN"):
        # 加载环境变量
        load_dotenv()

        # 设置源语言和目标语言
        self.source_lang = source_lang
        self.target_lang = "en-US" if source_lang == "zh-CN" else "zh-CN"

        # 语言映射
        self.lang_map = {
            "zh-CN": {"name": "中文", "target": "英文"},
            "en-US": {"name": "英文", "target": "中文"}
        }

        # 设置 Azure Speech 配置
        self.speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv('AZURE_SPEECH_KEY'),
            region=os.getenv('AZURE_SPEECH_REGION')
        )
        self.speech_config.speech_recognition_language = self.source_lang

        # 设置 VAD 超时时间
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "300")

        # 设置日志文件
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_LogFilename, "log.txt")

        # 设置 OpenAI 配置
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE')
        )

        # 创建识别结果队列
        self.result_queue = queue.Queue()

        # 创建音频流和识别器
        self.push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,
                bits_per_sample=16,
                channels=1
            )
        )
        self.speech_recognizer = self.create_recognizer(self.push_stream)

        # 启动持续识别
        self.speech_recognizer.start_continuous_recognition_async()

        print(f"当前设置: 识别{self.lang_map[self.source_lang]['name']}, "
              f"翻译为{self.lang_map[self.source_lang]['target']}")

        # 设置音频输出格式为 16kHz 16bit 单声道 PCM
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm)

        # 创建语音合成器
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None
        )

        # 语音配置映射
        self.voice_map = {
            "zh-CN": "zh-CN-XiaoxiaoNeural",
            "en-US": "en-US-JennyNeural"
        }

    def __del__(self):
        """析构函数，确保资源被正确释放"""
        try:
            if hasattr(self, 'speech_recognizer'):
                self.speech_recognizer.stop_continuous_recognition()
            if hasattr(self, 'push_stream'):
                self.push_stream.close()
        except Exception as e:
            print(f"清理资源时出错: {e}")

    def switch_language(self):
        """切换语言和目标语言"""
        try:
            # 停止当前识别
            self.speech_recognizer.stop_continuous_recognition()

            # 切换语言
            self.source_lang, self.target_lang = self.target_lang, self.source_lang
            self.speech_config.speech_recognition_language = self.source_lang

            # 重新创建识别器，使用相同的音频格式
            self.push_stream = speechsdk.audio.PushAudioInputStream(
                stream_format=speechsdk.audio.AudioStreamFormat(
                    samples_per_second=16000,
                    bits_per_sample=16,
                    channels=1
                )
            )
            self.speech_recognizer = self.create_recognizer(self.push_stream)
            self.speech_recognizer.start_continuous_recognition_async()

            print(f"语言已切换: 识别{self.lang_map[self.source_lang]['name']}, "
                  f"翻译为{self.lang_map[self.source_lang]['target']}")
        except Exception as e:
            print(f"切换语言时出错: {e}")

    def handle_recognized(self, evt):
        """处理识别到的结果"""
        print(evt)
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"识别到: {evt.result.text}")
            self.result_queue.put(evt.result.text)

    def handle_recognizing(self, evt):
        """处理识别中的结果"""
        print(evt)

    def handle_canceled(self, evt):
        """处理取消事件"""
        cancellation_details = evt.result.cancellation_details
        print(f"语音识别被取消: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"错误详情: {cancellation_details.error_details}")

    def create_recognizer(self, audio_stream):
        """创建语音识别器"""
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)

        # 不需要再次设置音频格式，因为已经在创建 push_stream 时设置
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        # 注册事件处理器
        speech_recognizer.session_started.connect(
            lambda evt: print('SESSION STARTED: {}'.format(evt)))
        speech_recognizer.recognizing.connect(self.handle_recognizing)
        speech_recognizer.recognized.connect(self.handle_recognized)
        speech_recognizer.canceled.connect(self.handle_canceled)

        return speech_recognizer

    def transcribe_audio(self, audio_data):
        """使用 Azure Speech 将音频转换为文本（流式识别）"""
        try:
            # 写入音频数据到流
            self.push_stream.write(audio_data.tobytes())

            # 等待识别结果
            try:
                result = self.result_queue.get(timeout=0.1)
                return result
            except queue.Empty:
                return None

        except Exception as e:
            print(f"音频识别错误: {e}")
            return None

    async def synthesize_speech(self, text, target_lang):
        """将文本转换为语音"""
        try:
            # 设置对应语言的语音
            self.speech_synthesizer.speech_synthesis_voice_name = self.voice_map[target_lang]

            # 直接使用同步方式获取结果
            result = self.speech_synthesizer.speak_text(text)

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # 获取音频数据
                print(f"合成音频大小: {len(result.audio_data)} bytes")
                audio_data = np.frombuffer(result.audio_data, dtype=np.int16)
                return audio_data
            else:
                print(f"语音合成失败: {result.reason}")
                return None

        except Exception as e:
            print(f"语音合成错误: {e}")
            return None

    async def translate_text(self, text):
        """使用 OpenAI GPT-4o-mini 进行翻译并合成语音"""
        try:
            target_lang_name = self.lang_map[self.source_lang]['target']
            prompt = f"请将以下{self.lang_map[self.source_lang]['name']}翻译成{target_lang_name}，只返回翻译结果：\n{text}"

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个专业的翻译助手，请直接提供翻译结果，不要添加任何解释。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            translated_text = response.choices[0].message.content.strip()
            print(f"翻译结果: {translated_text}")

            # 合成语音
            audio_data = await self.synthesize_speech(translated_text, self.target_lang)
            print(f"合成音频大小: {len(audio_data)} bytes")
            return translated_text, audio_data

        except Exception as e:
            print(f"翻译时出错: {e}")
            return None, None
