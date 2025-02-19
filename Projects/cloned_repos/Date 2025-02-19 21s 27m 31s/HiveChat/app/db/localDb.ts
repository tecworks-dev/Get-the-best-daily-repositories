import Dexie, { Table } from 'dexie';
import { Message } from './schema';


export class ChatDexie extends Dexie {
  messages!: Table<Message>;
  constructor() {
    super('ChatDB');
    this.version(1).stores({
      messages: '++id, chatId, createdAt',
    });
  }
}

export const localDb = new ChatDexie();