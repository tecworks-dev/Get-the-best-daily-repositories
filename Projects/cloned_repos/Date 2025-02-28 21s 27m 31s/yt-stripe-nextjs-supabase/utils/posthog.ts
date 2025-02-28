'use client'

import posthog from 'posthog-js'

let posthogClient: typeof posthog | null = null

// Initialize PostHog with better error handling
export const initPostHog = () => {
  if (posthogClient) return posthogClient
  
  try {
    if (typeof window !== 'undefined' && process.env.NEXT_PUBLIC_POSTHOG_KEY) {
      posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY, {
        api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://app.posthog.com',
        loaded: (posthog) => {
          if (process.env.NODE_ENV === 'development') posthog.debug(false)
        },
        autocapture: false, // Disable autocapture for better performance
        capture_pageview: false, // We'll handle this manually
        persistence: 'localStorage',
        cross_subdomain_cookie: false
      })
      posthogClient = posthog
    }
  } catch (error) {
    // Fail silently in production, log in development
    if (process.env.NODE_ENV === 'development') {
      console.error('PostHog initialization error:', error)
    }
  }
  
  return posthogClient
}