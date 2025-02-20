import { drizzle as neon } from 'drizzle-orm/neon-http';
import { drizzle } from 'drizzle-orm/postgres-js';

import { llmModels, users, llmSettingsTable, appSettings } from './schema';

const getDbInstance = () => {
  if (process.env.VERCEL) {
    return neon(process.env.DATABASE_URL!,
      { schema: { users, llmModels, llmSettingsTable, appSettings } });
  } else {
    return drizzle(process.env.DATABASE_URL!,
      { schema: { users, llmModels, llmSettingsTable, appSettings } });
  }
}

export const db = getDbInstance();
// export const db = drizzle(process.env.DATABASE_URL!, { schema: { users, llmModels, llmSettingsTable } });