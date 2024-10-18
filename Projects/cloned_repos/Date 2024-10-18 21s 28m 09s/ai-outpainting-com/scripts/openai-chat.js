const openai  = require("@ai-sdk/openai")
const { generateText } = require('ai');
const doubao = openai.createOpenAI({
  baseURL: "https://ark.cn-beijing.volces.com/api/v3",
  apiKey: "7da1e475-xxxxxxx",
  compatibility: 'compatible',
})
const MODEL_DOUBAO_LITE_4K = "ep-20240906111329-q22z4"
async function doubaoTranslate(prompt) {
  const { text } = await generateText({
    model: doubao(MODEL_DOUBAO_LITE_4K),
    prompt: prompt,
  });
  return {
    choices: [
      {
        message: {
          content: text
        }
      }
    ]
  };
}
async function translate(prompt) {
  return await  openAIChat(prompt)
}

function kimiAIChat(prompt) {
  const requestData = {
    model: 'moonshot-v1-8k',
    messages: [{ role: 'user', content: prompt }]
  }
  const requestStr = JSON.stringify(requestData)
  // console.log('openai request:', requestStr)
  const url = 'https://api.moonshot.cn/v1/chat/completions'
  // 翻译专用 key
  const key = 'sk-svcp0mYFph83hbgaPUD8a1wkuWAL6wXpkFR8ITrxJgcVglD8'
  return fetch(url, {
    method: 'POST', // Assuming it's a POST request
    headers: {
      'Content-Type': 'application/json',
      'Authorization': key
    },
    body: requestStr
  }).then((response) => response.json())
}

function openAIChat(prompt, model = 'gpt-3.5-turbo-0125') {
  const requestData = {
    model,
    messages: [{ role: 'user', content: prompt }]
  }
  const requestStr = JSON.stringify(requestData)
  // console.log('openai request:', requestStr)
  const url = 'https://api.inferkit.ai/v1/chat/completions'
  const key = 'Bearer sk-xxxxxxx'
  return fetch(url, {
    method: 'POST', // Assuming it's a POST request
    headers: {
      'Content-Type': 'application/json',
      'Authorization': key
    },
    body: requestStr
  }).then((response) => response.json())
}

async function openAIChat4(prompt, model = 'gpt-4-1106-preview') {
  const result = await openAIChat(prompt)
  if(result.choices.length > 0){
    return result.choices[0].message.content
  }else{
    console.log("openai返回结果：",JSON.stringify(result))
    throw new Error('openai返回结果为空')
  }
}

function qroqChat(prompt) {
  const requestData = {
    model: 'mixtral-8x7b-32768',
    messages: [{ role: 'user', content: prompt }]
  }
  const requestStr = JSON.stringify(requestData)
  // console.log('openai request:', requestStr)
  const url = 'https://api.groq.com/openai/v1/chat/completions'
  const key = 'Bearer gsk_xxxxx'
  return fetch(url, {
    method: 'POST', // Assuming it's a POST request
    headers: {
      'Content-Type': 'application/json',
      'Authorization': key
    },
    body: requestStr
  }).then((response) => response.json())
}

async function qWenOpenAIChat(prompt) {
  const GENERATION_URL =
    'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
  const API_KEY = 'sk-xxxxxx'

  const requestData = {
    model: 'qwen-turbo',
    input: { messages: [{ role: 'user', content: prompt }] },
    parameters: {}
  }
  const requestStr = JSON.stringify(requestData)
  const result = await fetch(GENERATION_URL, {
    method: 'POST', // Assuming it's a POST request
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`
    },
    body: requestStr
  }).then((response) => response.json())
  // 构造成openai返回的格式
  return { choices: [{ message: { content: result.output.text } }] }
}

qWenOpenAIChat('翻译这一句：hello world 为中文')

module.exports = {
  translate,
  openAIChat4
}