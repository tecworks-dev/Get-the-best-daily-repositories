# 🚀 JARVIS 2.0

---

# 🤖 Jarvis AI Assistant

Welcome to the **Jarvis AI Assistant** project! 🎙️ This AI-powered assistant can perform various tasks such as **providing weather reports 🌦️, summarizing news 📰, sending emails 📧**, and more, all through **voice commands**. Below, you'll find detailed instructions on how to set up, use, and interact with this assistant. 🎧

---

## 🌟 Features

✅ **Voice Activation**: Say **"Hey Jarvis"** to activate listening mode. 🎤\
✅ **Speech Recognition**: Recognizes and processes user commands via speech input. 🗣️\
✅ **AI Responses**: Provides responses using AI-generated **text-to-speech** output. 🎶\
✅ **Task Execution**: Handles multiple tasks, including:

- 📧 **Sending emails**
- 🌦️ **Summarizing weather reports**
- 📰 **Reading news headlines**
- 🖼️ **Image generation**
- 🏦 **Database functions**
- 📱 **Phone call automation using ADB**
- 🤖 **AI-based task execution**
- 📡 **Automate websites & applications**
- 🧠 **Retrieval-Augmented Generation (RAG) for knowledge-based interactions**
- ✅ **Timeout Handling**: Automatically deactivates listening mode after **5 minutes** of inactivity. ⏳
- ✅ **Automatic Input Processing**: If no "stop" command is detected within **60 seconds**, input is finalized and sent to the AI model for processing. ⚙️
- ✅ **Multiple Function Calls**: Call **multiple functions simultaneously**, even if their inputs and outputs are unrelated. 🔄

---

## 📌 Prerequisites

Before running the project, ensure you have the following installed:

✅ **Python 3.9 or later** 🐍\
✅ Required libraries (listed in `requirements.txt`) 📜

### 🛠️ Configuration

1. **Create a ************`.env`************ file** in the root directory of the project.

2. **Add your API keys and other configuration variables** to the `.env` file:

   ```dotenv
   Weather_api=your_weather_api_key
   News_api=your_news_api_key
   Sender_email=your_email
   Receiver_email=subject_email
   Password_email=email_password
   ```

3. **Setup API Keys & Passwords**:

   - [🌩️ WEATHER API](https://rapidapi.com/weatherapi/api/weatherapi-com) - Get weather data.
   - [📰 NEWS API](https://newsapi.org) - Fetch latest news headlines.
   - [📧 GMAIL PASSWORD](https://myaccount.google.com/apppasswords) - Generate an app password for sending emails.
   - [🧠 OLLAMA](https://ollama.com) - Download **Granite3.1-Dense:2b/8b** models from Ollama.
   - [🔮 GEMINI AI](https://ai.google.dev/) - API access for function execution.

## Directory structure 
```
├── DATA
│   ├── KNOWLEDGEBASE
│   │   └── disaster_data_converted.md
│   ├── RAWKNOWLEDGEBASE
│   │   └── disaster_data.pdf
│   ├── email_schema.py
│   ├── msg.py
│   ├── phone_details.py
│   ├── samples
│   │   ├── share_func.py
│   │   ├── tools.json
│   │   └── tools_new.json
│   └── tools.py
├── device_ips.txt
├── main.py
├── readme.md
├── requirements.txt
└── src
    ├── BRAIN
    │   ├── RAG.py
    │   ├── func_call.py
    │   ├── gemini_llm.py
    │   ├── lm_ai.py
    │   └── text_to_info.py
    ├── CONVERSATION
    │   ├── speech_to_text.py
    │   ├── t_s.py
    │   ├── test_speech.py
    │   └── text_to_speech.py
    ├── FUNCTION
    │   ├── Email_send.py
    │   ├── adb_connect.bat
    │   ├── adb_connect.sh
    │   ├── app_op.py
    │   ├── get_env.py
    │   ├── greet_time.py
    │   ├── incog.py
    │   ├── internet_search.py
    │   ├── link_op.py
    │   ├── news.py
    │   ├── phone_call.py
    │   ├── random_respon.py
    │   ├── run_function.py
    │   ├── weather.py
    │   └── youtube_downloader.py
    ├── KEYBOARD
    │   ├── key_lst.py
    │   └── key_prs_lst.py
    └── VISION
        └── eye.py

11 directories, 40 files
```
---

## 💻 Installation

### 1️⃣ **Clone the Repository**

```bash
 git clone https://github.com/ganeshnikhil/J.A.R.V.I.S.2.0.git
 cd J.A.R.V.I.S.2.0
```

### 2️⃣ **Install Dependencies**

```bash
 pip install -r requirements.txt
```

---

## 🚀 Running the Application

### **Start the Program**

```bash
 python main.py
```

📢 **Initial Interaction**:

```plaintext
[= =] Say 'hey jarvis' to activate, and 'stop' to deactivate. Say 'exit' to quit.
```

---

## 🔄 **Function Calling Methods**

### 🔹 **Primary: Gemini AI-Based Function Execution**

🚀 Transitioned to **Gemini AI-powered function calling**, allowing multiple **function calls simultaneously** for better efficiency! ⚙️ If Gemini AI fails to generate function calls, the system automatically falls back to an **Ollama-based model** for reliable execution. 

🔹 **AI Model Used**: **Gemini AI** 🧠\
✅ Higher accuracy ✅ Structured data processing ✅ Reliable AI-driven interactions

📌 **Command Parsing** 📜

```python
response = gemini_generate_function_call(command)
response_dic = parse_tool_call(response)
```

📌 **Dynamic Function Execution** 🔄

```python
if response_dic:
    func_name = response_dic["name"]
    response = execute_function_call(response_dic)
```

📌 **Error Handling & Fallback to Ollama** 🛑

```python
try:
    response = execute_function_call(response_dic)
except Exception as e:
    print(f"Error in Gemini AI function execution: {e}")
    print("Falling back to Ollama-based function execution...")
    response = ollama_generate_function_call(command)
```

📌 **Retry Mechanism** 🔄

```python
def send_to_ai_with_retry(prompt, retries=3, delay=2):
    for _ in range(retries):
        try:
            return send_to_gemini(prompt)
        except Exception:
            time.sleep(delay)
    print("Gemini AI is not responding. Switching to Ollama...")
    return send_to_ollama(prompt)
```

---

## 📖 **RAG-Based Knowledge System**

💡 **Retrieval-Augmented Generation (RAG)** dynamically loads relevant markdown-based knowledge files based on the queried topic, **reducing hallucinations and improving response accuracy**.

---

## 📱 **ADB Integration for Phone Automation**

🔹 Integrated **Android Debug Bridge (ADB)** to enable **voice-controlled phone automation**! 🎙️

✅ **Make phone calls** ☎️\
✅ **Open apps & toggle settings** 📲\
✅ **Access phone data & remote operations** 🛠️

### **Setting Up ADB**

📌 **Windows**

```powershell
winget install --id=Google.AndroidSDKPlatformTools -e
```

📌 **Linux**

```bash
sudo apt install adb
```

📌 **Mac**

```bash
brew install android-platform-tools
```

---

## 🔮 **Future Enhancements**

✨ **Deeper mobile integration** 📱\
✨ **Advanced AI-driven automation** 🤖\
✨ **Improved NLP-based command execution** 🧠\
✨ **Multi-modal interactions (text + voice + image)** 🖼️

🚀 **Stay tuned for future updates!** 🔥

---
```
RPM	   TPM	    RPD
Gemini 2.0 Flash	15	1,000,000	1,500
Gemini 2.0 Flash-Lite Preview	30	1,000,000	1,500
Gemini 2.0 Pro Experimental 02-05	2	1,000,000	50
Gemini 2.0 Flash Thinking Experimental 01-21	10	4,000,000	1,500
Gemini 1.5 Flash	15	1,000,000	1,500
Gemini 1.5 Flash-8B	15	1,000,000	1,500
Gemini 1.5 Pro	2	32,000	50
Imagen 3	--	--	--
```
