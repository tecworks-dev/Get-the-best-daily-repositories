import axios from "axios";
import instance from "./index";

// 根据环境配置baseURL
const API_BASE_URL =
  process.env.VUE_APP_API_BASE_URL || "http://101.6.21.31:10003/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: false, // 跨域访问时关闭
});

export interface FuzzingConfig {
  model: string;
  seedInputType: string;
  questionInputType: string;
  seedContent?: string;
  questionContent?: string;
  questionFile?: string;
  mutators: string[];
  maxIterations: number;
  maxSuccesses: number;
  task?: string;
}

export interface LogResponse {
  logs: string[];
}

export interface QuestionFile {
  name: string;
}

export interface QuestionFileContent {
  content: string[];
}

export interface Experiment {
  id: string;
  timestamp: string;
  has_seed_flow: boolean;
  has_summary: boolean;
}

export interface SeedFlowData {
  [key: string]: {
    id: string;
    content: string;
    parent_id: string | null;
    mutation_type: string | null;
    creation_time: string;
    depth: number;
    children: string[];
    stats: {
      uses: number;
      successes: number;
      total_trials: number;
    };
    metadata: Record<string, any>;
  };
}

// 添加新的类型定义
export interface ExperimentLogs {
  main: string[];
  mutation: string[];
  jailbreak: string[];
  error: string[];
}

export interface ExperimentResults {
  experiment_summary: string;
  all_results: any;
  question_success_details?: {
    [key: string]: Array<{
      iteration: number;
      response_en: string;
      response_zh: string;
    }>;
  };
}

// 获取可用的问题文件列表
export const getQuestionFiles = async () => {
  try {
    const response = await api.get<{ files: string[] }>("/question-files");
    return response.data.files;
  } catch (error) {
    console.error("Error fetching question files:", error);
    throw new Error("Failed to get question files");
  }
};

// 获取问题文件内容
export const getQuestionFileContent = async (filename: string) => {
  try {
    const response = await api.get<QuestionFileContent>(
      "/question-file-content",
      {
        params: { filename },
      }
    );
    return response.data.content;
  } catch (error) {
    throw new Error("Failed to get file content");
  }
};

export const startFuzzing = async (config: FuzzingConfig) => {
  try {
    const response = await api.post("/start", config);
    if (response.data.error) {
      throw new Error(response.data.error);
    }
    return response.data;
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`Failed to start fuzzing: ${error.message}`);
    }
    throw new Error("Failed to start fuzzing");
  }
};

export const stopFuzzing = async () => {
  try {
    const response = await api.post("/stop");
    return response.data;
  } catch (error) {
    throw new Error("Failed to stop fuzzing");
  }
};

export const getFuzzingStatus = async (sessionId: string) => {
  try {
    const response = await api.get("/status", {
      params: { session_id: sessionId },
    });
    return response.data;
  } catch (error) {
    throw new Error("Failed to get status");
  }
};

export const getLogs = async (
  sessionId: string,
  type = "main",
  maxLines = 0
) => {
  try {
    const response = await api.get<LogResponse>("/logs", {
      params: {
        session_id: sessionId,
        type,
        max_lines: maxLines,
      },
    });
    return response.data;
  } catch (error) {
    throw new Error("Failed to get logs");
  }
};

export const downloadResults = async (sessionId: string) => {
  try {
    const response = await api.get("/download-results", {
      params: { session_id: sessionId },
    });
    return response.data;
  } catch (error) {
    console.error("API Error in downloadResults:", error);
    throw error;
  }
};

export const clearSession = async () => {
  try {
    const response = await api.post("/clear-session");
    return response.data;
  } catch (error) {
    throw new Error("Failed to clear session");
  }
};

export const getExperiments = async () => {
  try {
    const response = await api.get<{ experiments: Experiment[] }>(
      "/experiments"
    );
    return response.data.experiments;
  } catch (error) {
    console.error("Error fetching experiments:", error);
    throw new Error("Failed to get experiments");
  }
};

export const getSeedFlow = async (experimentId: string) => {
  try {
    const response = await api.get<{ seed_flow: SeedFlowData }>(
      `/experiments/${experimentId}/seed-flow`
    );
    return response.data.seed_flow;
  } catch (error) {
    console.error("Error fetching seed flow:", error);
    throw new Error("Failed to get seed flow data");
  }
};

export const getExperimentLogs = async (
  experimentId: string
): Promise<ExperimentLogs> => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/experiments/${experimentId}/logs`
    );
    return response.data;
  } catch (error) {
    console.error("Failed to fetch experiment logs:", error);
    // 返回空日志作为默认值
    return {
      main: [],
      mutation: [],
      jailbreak: [],
      error: [],
    };
  }
};

export const getExperimentResults = async (
  experimentId: string
): Promise<ExperimentResults> => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/experiments/${experimentId}/results`
    );
    return response.data;
  } catch (error) {
    console.error("Failed to fetch experiment results:", error);
    // 返回默认结果
    return {
      experiment_summary: "暂无测试结果",
      all_results: null,
      question_success_details: {},
    };
  }
};
