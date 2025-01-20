import { createClient } from '@supabase/supabase-js';
import { environment } from './environment';

if (!environment.supabaseUrl || !environment.supabaseKey) {
  throw new Error('Missing Supabase configuration. Check SUPABASE_URL and SUPABASE_SERVICE_KEY in .env');
}

// Create Supabase client with service key for admin access
export const supabase = createClient(
  environment.supabaseUrl as string,
  environment.supabaseKey as string,
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  }
);

// Helper function to get Supabase client with user's JWT
export const getSupabaseClient = (jwt: string) => {
  if (!environment.supabaseUrl || !environment.supabaseKey) {
    throw new Error('Missing Supabase configuration');
  }

  const client = createClient(
    environment.supabaseUrl as string,
    environment.supabaseKey as string,
    {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      },
      global: {
        headers: {
          Authorization: `Bearer ${jwt}`
        }
      }
    }
  );
  
  return client;
}; 