export interface FetchOptions {
    timeout: number;
    waitUntil: 'load' | 'domcontentloaded' | 'networkidle' | 'commit';
    extractContent: boolean;
    maxLength: number;
    returnHtml: boolean;
  }
  
  export interface FetchResult {
    success: boolean;
    content: string;
    error?: string;
    index?: number;
  }