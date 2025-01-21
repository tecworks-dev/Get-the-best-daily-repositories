declare global {
  namespace NodeJS {
    interface ProcessEnv {
      PORT: string;
      NODE_ENV: 'development' | 'production' | 'test';
      OPENAI_API_KEY: string;
      SUPABASE_URL: string;
      SUPABASE_SERVICE_KEY: string;
    }
  }
}

export {}; 