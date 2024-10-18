
// 根据语言和id获取翻译后的数据
import path from 'path'
import fs from 'fs'

export function getTranslationById(id:string,lang:string){
  const filePath = path.join("translations", lang,"messages.json")
  const content = fs.readFileSync(filePath,"utf-8")
  const messages = JSON.parse(content)
  return messages[id]?.translation
}
