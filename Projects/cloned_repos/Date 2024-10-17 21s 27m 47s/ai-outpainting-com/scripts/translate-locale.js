const fs = require('fs').promises;
const path = require('path');
const localeNames = require(path.join(__dirname, '../framework/locale/localeConfig'));
const directoryPath = path.join(__dirname, '../translations');
const { translate } = require('./openai-chat');

async function processLanguage(file) {
    const language = file.split('.')[0];
    if (language === 'en') {
        return;
    }
    console.log("正在处理语言：" + language + "," + localeNames[language]);

    const data = await fs.readFile(path.join(directoryPath, file + '/messages.json'), 'utf8');
    const json = JSON.parse(data);
    const keys = Object.keys(json);

    const needTranslateKeys = {};
    keys.forEach(function (key) {
        if (Object.keys(needTranslateKeys).length >= 50) {
            console.log("当前语言文件需要翻译的数量过多，请多次执行，避免有遗漏");
        } else if (!json[key]['translation'] || json[key]['translation'] === '') {
            needTranslateKeys[key] = json[key]['message'];
        }
    });

    if (Object.keys(needTranslateKeys).length === 0) {
        console.log("当前语言文件已翻译完成，无需翻译");
        return;
    }

    console.log(`当前语言:${language},需要翻译的key数量:${Object.keys(needTranslateKeys).length}`);

    const prompt = `
        - 你是一个擅长数据处理和多语言翻译的AI专家，具备高效处理JSON数据和灵活应对多种语言需求的能力。 
        - 翻译考虑到专业术语和正式风格，适用于正式文档和官方交流。
        - 翻译的结果输出为JSON内容key保持不变，直接输出json内容不要加\`\`\`json\`\`\`标签。，不要做任何解释
        - 保证json格式准确性，确保key与内容成对出现。
        - 翻译考虑使用当地的习惯用语，而不是简单的文字翻译，了解原始文字的意境找到当地的表达方式进行翻译
        - 翻译目标语言为：${localeNames[language]}
        - 不要做任何解释，直接输出json内容，也不要输出\`\`\`json\`\`\`标签
        - 输入JSON数据：
            ${JSON.stringify(needTranslateKeys, null, 2)}
    `;

    let msg = await translate(prompt);
    console.log("openai返回值:", JSON.stringify(msg));
    msg = msg.choices[0].message.content;

    const safeJSONParse = (str) => {
        try {
            return JSON.parse(str);
        } catch (e) {
            console.error("JSON解析失败，尝试修复");
            str = str.replace(/[\u0000-\u001F]+/g, "");
            str = str.replace(/(["\]}])([^,\]}])/g, "$1,$2");
            str = str.replace(/([\[{])\s*,/g, "$1");
            str = str.replace(/,\s*([\]}])/g, "$1");
            try {
                return JSON.parse(str);
            } catch (e) {
                console.error("修复后仍无法解析JSON", e);
                return null;
            }
        }
    };

    msg = safeJSONParse(msg);

    if (msg === null) {
        console.error("无法解析返回的JSON数据");
    } else {
        keys.forEach(function (key) {
            if (msg[key]) {
                json[key]['translation'] = msg[key];
            }
        });
        const jsonStr = JSON.stringify(json, null, 2);
        await fs.writeFile(path.join(directoryPath, file + '/messages.json'), jsonStr, 'utf8');
    }
}

async function processLanguagesInQueue(languages, concurrency = 3) {
    const queue = [...languages];
    const inProgress = new Set();
    const results = [];

    async function processNext() {
        if (queue.length === 0) return;
        const language = queue.shift();
        inProgress.add(language);

        try {
            await processLanguage(language);
            results.push(`${language} 处理完成`);
        } catch (error) {
            results.push(`${language} 处理失败: ${error.message}`);
        } finally {
            inProgress.delete(language);
            if (queue.length > 0) {
                await processNext();
            }
        }
    }

    const workers = Array(Math.min(concurrency, languages.length))
        .fill()
        .map(() => processNext());

    await Promise.all(workers);
    return results;
}

async function main() {
    try {
        const files = await fs.readdir(directoryPath);
        const languagesToProcess = files.filter(file => file !== 'en');
        const results = await processLanguagesInQueue(languagesToProcess);
        console.log("处理结果:", results.join('\n'));
    } catch (err) {
        console.error('Error:', err);
    }
}

main();





