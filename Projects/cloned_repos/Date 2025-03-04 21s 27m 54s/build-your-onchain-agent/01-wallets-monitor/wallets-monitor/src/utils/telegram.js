import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config();

// Sends a message to Telegram chat with optional reply functionality
export async function sendTelegramMessage(message, replyToMessageId = null) {
  const botToken = process.env.TELEGRAM_TOKEN;
  const channelId = process.env.TELEGRAM_CHAT_ID;
  const url = `https://api.telegram.org/bot${botToken}/sendMessage`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        chat_id: channelId,
        text: message,
        parse_mode: 'HTML',
        disable_web_page_preview: true,
        reply_to_message_id: replyToMessageId
      }),
    });

    const data = await response.json();
    // console.log('Telegram response:', data);
    
    if (!data.ok || data.description?.includes('Unknown error')) {
      throw new Error(`Telegram API error: ${data.description || 'Unknown error'}`);
    }
    
    return data;
  } catch (error) {
    console.error('Error sending Telegram message:', error);
    throw error; 
  }
}

