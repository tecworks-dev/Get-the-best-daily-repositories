import { searchTwitter, getUserTimeline } from './tweetApi.js';
import { sendTelegramMessage } from './telegram.js';
import OpenAI from "openai";
import dotenv from 'dotenv';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com/v1",
});

// Summarizes tweets related to a token from both account and search results
async function sumTweets(tokenInfo) {
  const { symbol, address, twitter } = tokenInfo;
  
  let account_tweets = [];
  let search_tweets = [];
  
  // Get tweets from Twitter account
  if (twitter && (twitter.includes('x.com/') || twitter.includes('twitter.com/'))) {
    const urlParts = twitter.split('/');
    // Exclude special links
    if (!twitter.includes('/communities/') && !twitter.includes('/search?') && !twitter.includes('/status/')) {
      let screenname = urlParts[urlParts.length - 1].split('?')[0];
      
      const timelineResult = await getUserTimeline(screenname);
      if (timelineResult) account_tweets = timelineResult;
      else console.log('Failed to fetch user tweets:', screenname);
    }
  }
  
  // Search for tweets related to token address
  search_tweets = await searchTwitter(address);
  
  if (!search_tweets?.length) {
    console.log('No tweets found for address:', address);
    return `No tweet data found for ${symbol}(${address}).`;
  }
  
  // Analyze tweets
  const search_summary = await genSum(symbol, search_tweets, 'search');
  
  let account_summary = "";
  if (account_tweets?.tweets?.length > 0) {
    account_summary = await genSum(symbol, account_tweets, 'account');
  }
  
  if (!search_summary && !account_summary) {
    console.log(`Unable to generate tweet analysis summary for ${symbol}.`);
    return null;
  }
  
  return { search_summary, account_summary };
}

// Generates a summary of tweets using AI
async function genSum(symbol, tweets, type = 'search') {
  try {
    let tweetData = [];
    let promptPrefix = '';
    let promptSuffix = '';
    
    if (type === 'account') {
      promptPrefix = `请总结关于 ${symbol} 的账号推文:`;
      promptSuffix = `提供简短的要点总结。保持简洁直接,去除所有不必要的词语。`;
      
      // Process account tweets format
      tweetData = tweets.tweets.map((tweet, index) => `
Tweet ${index + 1}:
Content: ${tweet.text}
Time: ${tweet.created_at}
Engagement: ${tweet.views} views / ${tweet.favorites} likes 
---`);
    } else {
      // Search tweets
      promptPrefix = `请总结关于 ${symbol} 的搜索推文:`;
      promptSuffix = `提供关于叙事观点和风险内容的极简要点总结。不总结主观价格预测和个人收益的内容。保持简洁直接,去除所有不必要的词语。格式如下：
- 叙事观点：
- 风险内容：`;
      
      // Process search tweets format
      tweetData = tweets.map((tweet, index) => `
Tweet ${index + 1}:
Content: ${tweet.text}
Time: ${tweet.created_at}
Author: ${tweet.author.name} (@${tweet.author.screen_name})
Followers: ${tweet.author.followers_count}
Engagement: ${tweet.views} views / ${tweet.favorites} likes 
---`);
    }
    
    const prompt = `${promptPrefix}

${tweetData.join('\n')}

${promptSuffix}`;

    const response = await openai.chat.completions.create({
      model: "deepseek-chat",
      messages: [
        { role: "system", content: "You are a helpful assistant that analyzes cryptocurrency Twitter data." },
        { role: "user", content: prompt }
      ],
      temperature: 1.0,
      max_tokens: 3000
    });

    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error generating Twitter summary:", error);
    return "Failed to generate summary due to an error.";
  }
}

// Sends the tweet summary to Telegram as a reply to a message
export async function sendSumMessage(tokenInfo, replyToMessageId) {
  const summaryResult = await sumTweets(tokenInfo);
  if (!summaryResult) {
    console.log(`Unable to get tweet summary for ${tokenInfo.symbol}`);
    return;
  }
  
  const { search_summary, account_summary } = summaryResult;
  
  let message = `\u{1F49B}${tokenInfo.symbol} tweets summary:\n`;
  
  if (account_summary) {
    // Format line breaks and spaces, replace multiple line breaks with a single one
    const formattedAccountSummary = account_summary
      .replace(/\n\s*\n/g, '\n')  
      .trim();  
    message += `<blockquote>${formattedAccountSummary}</blockquote>\n\n`;
  }
  
  if (search_summary) {
    message += `\u{1F49B}Searched tweets summary:\n<blockquote>${search_summary}</blockquote>`;
  }
  
  const tgResponse = await sendTelegramMessage(message, replyToMessageId);
  
  return tgResponse;
}
