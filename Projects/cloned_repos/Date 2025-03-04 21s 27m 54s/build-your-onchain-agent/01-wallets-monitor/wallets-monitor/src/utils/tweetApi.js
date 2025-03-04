import axios from 'axios';
import dotenv from 'dotenv';
import axiosRetry from 'axios-retry';

dotenv.config();

const RAPID_API_KEY = process.env.RAPID_API_KEY;
const TWITTER_API_HOST = 'twitter-api45.p.rapidapi.com';
// https://rapidapi.com/alexanderxbx/api/twitter-api45

axiosRetry(axios, {
  retries: 3,
  retryDelay: (retryCount) => {
    return retryCount * 1000; // Increasing delay for each retry
  },
  retryCondition: (error) => {
    // Only retry on network errors or 5xx errors
    return axiosRetry.isNetworkOrIdempotentRequestError(error) || 
           (error.response && error.response.status >= 500);
  }
});

// Searches for content on Twitter with specified query and search type
export async function searchTwitter(query, searchType = 'Top') {
  const options = {
    method: 'GET',
    url: 'https://twitter-api45.p.rapidapi.com/search.php',
    params: {
      query,
      search_type: searchType
    },
    headers: {
      'x-rapidapi-key': RAPID_API_KEY,
      'x-rapidapi-host': TWITTER_API_HOST
    }
  };

  const response = await axios.request(options).catch(error => {
    console.error('Twitter API Error:', error.message);
    throw error;
  });

  if (!response || !response.data) {
    console.error('Twitter API Search Error: No response data');
    return null;
  }

  // Extract and process tweet data
  if (!response.data?.timeline) {
    console.error('Twitter API Search Error: No tweet data');
    return [];
  }

  return response.data.timeline.map(tweet => ({
    text: tweet.text,                    // Tweet content
    created_at: new Date(tweet.created_at).toLocaleString('en-US', {hour12: false}) + ' UTC',  // Published time, format: 2025/2/25 14:41:14 UTC
    
    // Engagement data
    views: tweet.views,                  // View count
    favorites: tweet.favorites,          // Like count
    retweets: tweet.retweets,           // Retweet count
    replies: tweet.replies,              // Reply count
    
    // Author information
    author: {
      name: tweet.user_info.name,
      screen_name: tweet.user_info.screen_name,
      followers_count: tweet.user_info.followers_count,
      description: tweet.user_info.description
    }
  }));
}

// Retrieves a Twitter user's timeline by screen name
export async function getUserTimeline(screenname) {
  const options = {
    method: 'GET',
    url: 'https://twitter-api45.p.rapidapi.com/timeline.php',
    params: {
      screenname
    },
    headers: {
      'x-rapidapi-key': RAPID_API_KEY,
      'x-rapidapi-host': TWITTER_API_HOST
    }
  };

  const response = await axios.request(options).catch(error => {
    console.error('Twitter API Error:', error.message);
    throw error;
  });

  if (!response || !response.data) {
    console.error('Twitter API Timeline Error: No response data');
    return null;
  }
  
  // Organize data structure
  const result = {
    user: {
      name: response.data.user?.name,
      screen_name: screenname,
      verified: response.data.user?.blue_verified,
      description: response.data.user?.desc,
      followers_count: response.data.user?.sub_count
    },
    tweets: []
  };
  
  // Add pinned tweet (if exists)
  if (response.data.pinned) {
    result.tweets.push({
      text: response.data.pinned.text,
      created_at: new Date(response.data.pinned.created_at).toLocaleString('en-US', {hour12: false}) + ' UTC', 
      views: response.data.pinned.views,
      favorites: response.data.pinned.favorites,
      retweets: response.data.pinned.retweets,
      replies: response.data.pinned.replies,
      isPinned: true
    });
  }
  
  // Add timeline tweets
  if (response.data.timeline && Array.isArray(response.data.timeline)) {
    response.data.timeline.forEach(tweet => {
      result.tweets.push({
        text: tweet.text,
        created_at: new Date(tweet.created_at).toLocaleString('en-US', {hour12: false}) + ' UTC', 
        views: tweet.views,
        favorites: tweet.favorites,
        retweets: tweet.retweets,
        replies: tweet.replies,
        isPinned: false
      });
    });
  }
  
  return result;
}

