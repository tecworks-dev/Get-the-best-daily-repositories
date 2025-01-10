import { Database } from 'duckdb-async';

const db = await Database.create(':memory:');
