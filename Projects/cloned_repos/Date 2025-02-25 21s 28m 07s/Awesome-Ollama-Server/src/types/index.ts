export interface OllamaService {
  server: string;
  models: string[];
  tps: number;
  lastUpdate: string;
}

export type SortField = 'tps' | 'lastUpdate';
export type SortOrder = 'asc' | 'desc'; 