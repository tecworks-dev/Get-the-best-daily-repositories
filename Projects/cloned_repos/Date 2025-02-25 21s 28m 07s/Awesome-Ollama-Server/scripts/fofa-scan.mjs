import { promises as fs } from 'fs';
import axios from 'axios';

export async function fofaScan(country = 'RU') {
    const searchQuery = `app="Ollama" && country="${country}"`;
    const encodedQuery = Buffer.from(searchQuery).toString('base64');
    const baseUrl = `https://fofa.info/result?qbase64=${encodedQuery}`;
    const userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36";

    try {
        const response = await axios.get(baseUrl, {
            headers: {
                'User-Agent': userAgent
            }
        });

        const HTML_START_TAG = 'hsxa-host"><a href="';
        const HTML_END_TAG = '"';
        const hosts = [];
        
        let currentIndex = 0;
        const responseText = response.data;
        
        while (true) {
            const startPosition = responseText.indexOf(HTML_START_TAG, currentIndex);
            if (startPosition === -1) break;
            
            const endPosition = responseText.indexOf(HTML_END_TAG, startPosition + HTML_START_TAG.length);
            if (endPosition === -1) break;
            
            hosts.push(responseText.slice(startPosition + HTML_START_TAG.length, endPosition));
            currentIndex = endPosition;
        }

        await fs.appendFile('result.txt', hosts.join('\n') + '\n', 'utf-8');
        hosts.forEach(host => console.log(host));
        console.log("已保存：result.txt", hosts.length, hosts);

        return { hosts };

    } catch (error) {
        const errorMessage = error.isAxiosError 
            ? `请求失败: ${error.message}`
            : `未知错误: ${error}`;
        
        console.error(errorMessage);
        return {
            hosts: [],
            errorMessage
        };
    }
}

// 自执行异步函数
(async () => {
    await fofaScan();
})();