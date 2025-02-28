import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

function getCorsHeaders(request: NextRequest) {
  const allowedOrigins = [
    'http://localhost:3000',
    "https://NextTemp.vercel.app"
  ];
  const origin = request.headers.get('origin') || '';
  
  return {
    'Access-Control-Allow-Origin': allowedOrigins.includes(origin) ? origin : allowedOrigins[0],
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, stripe-signature, x-client-info',
    'Access-Control-Max-Age': '86400', // 24 hours
  };
}

export function withCors(handler: Function) {
  return async function corsHandler(request: NextRequest) {
    // Handle preflight requests
    if (request.method === 'OPTIONS') {
      return NextResponse.json({}, { headers: getCorsHeaders(request) });
    }

    // Call the original handler
    const response = await handler(request);

    // Add CORS headers to the response
    Object.entries(getCorsHeaders(request)).forEach(([key, value]) => {
      response.headers.set(key, value);
    });

    return response;
  };
}