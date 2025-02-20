import 'dotenv/config';
import { llmModels } from '@/app/db/schema';
import { db } from './index';

import { modelList as OpenaiModels } from "@/app/db/data/models/openai";
import { modelList as ClaudeModels } from "@/app/db/data/models/claude";
import { modelList as GeminiModels } from "@/app/db/data/models/gemini";
import { modelList as MoonshotModels } from "@/app/db/data/models/moonshot";
import { modelList as QwenModels } from "@/app/db/data/models/qwen";
import { modelList as VolcengineModels } from "@/app/db/data/models/volcengine";
import { modelList as DeepseekModels } from "@/app/db/data/models/deepseek";
import { modelList as QianfanModels } from "@/app/db/data/models/qianfan";
import { modelList as SiliconflowModels } from "@/app/db/data/models/siliconflow";
import { modelList as OllamaModels } from "@/app/db/data/models/ollama";
const modelList = [
  ...OpenaiModels,
  ...ClaudeModels,
  ...GeminiModels,
  ...MoonshotModels,
  ...QwenModels,
  ...VolcengineModels,
  ...DeepseekModels,
  ...QianfanModels,
  ...SiliconflowModels,
  ...OllamaModels,
];

export async function initializeModels() {
  const count = await db.$count(llmModels);
  if (count > 0) {
    return;
  }
  const modelData = modelList.map((model) => ({
    name: model.id,
    displayName: model.displayName,
    maxTokens: model.maxTokens,
    supportVision: model.supportVision,
    selected: model.selected,
    providerId: model.provider.id,
    providerName: model.provider.providerName,
    type: model.type ?? 'default',
    createdAt: new Date(),
    updatedAt: new Date(),
  }));

  await db.insert(llmModels).values(modelData);
}

initializeModels().then(() => {
  console.log("Models initialized successfully.");
  process.exit(0); // 成功退出
}).catch((error) => {
  console.error("Error initializing models:", error);
  process.exit(1);
});