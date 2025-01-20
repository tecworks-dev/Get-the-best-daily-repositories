import dotenv from 'dotenv';

// Load test environment variables first
dotenv.config({ path: '.env.test' });

// Ensure we're using test environment
process.env.NODE_ENV = 'test';

import { supabase } from '../config/supabase';

// Mock auth middleware for tests
jest.mock('../middleware/auth', () => ({
  authMiddleware: (req: any, res: any, next: any) => {
    req.user = { id: process.env.TEST_USER_UUID };
    next();
  }
}));

// Add delay utility
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

beforeAll(async () => {
  if (!process.env.TEST_USER_UUID) {
    throw new Error('TEST_USER_UUID not found in environment variables');
  }

  // Clean up any existing test data
  await supabase
    .from('words')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000');

  await supabase
    .from('word_lists')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000');

  // Wait for deletions to complete
  await delay(1000);
});

afterAll(async () => {
  // Final cleanup
  await supabase
    .from('words')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000');

  await supabase
    .from('word_lists')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000');
}); 