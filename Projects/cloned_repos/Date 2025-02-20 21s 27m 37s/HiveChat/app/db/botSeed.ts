import 'dotenv/config';
import { bots } from '@/app/db/schema';
import { db } from './index';
import botsData from './data/bots';

export async function initializeBots() {
  const count = await db.$count(bots);
  if (count === 0) {
    await db.insert(bots).values(botsData);
  }
}

initializeBots().then(() => {
  console.log("Bots initialized successfully.");
  process.exit(0); // 成功退出
}).catch((error) => {
  console.error("Error initializing bots:", error);
  process.exit(1);
});