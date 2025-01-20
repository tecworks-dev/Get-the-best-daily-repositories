import rateLimit from 'express-rate-limit';

// Default rate limiter for general endpoints
export const defaultRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
});

// Stricter rate limiter for authentication-related endpoints
export const authRateLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 5, // Limit each IP to 5 requests per windowMs
  message: 'Too many authentication attempts, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});

// Rate limiter for learning-related endpoints (quiz, learn)
export const learningRateLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 200, // Limit each IP to 200 requests per hour
  message: 'You have exceeded the learning requests limit. Please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});

// Rate limiter for list management
export const listManagementRateLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 50, // Limit each IP to 50 list operations per hour
  message: 'Too many list operations. Please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});

// Rate limiter specifically for OpenAI-powered routes (learning and quiz)
export const openaiRateLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 100, // Limit each IP to 100 OpenAI requests per hour
  message: 'Too many AI-powered requests. Please try again in an hour.',
  standardHeaders: true,
  legacyHeaders: false,
}); 