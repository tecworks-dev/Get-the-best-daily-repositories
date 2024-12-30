<template>
  <div class="gpt-fuzzer-interface">
    <!-- 页面标题和说明 -->
    <h2>FuzzLLM</h2>
    <p>
      使用模糊测试对大语言模型进行安全性和鲁棒性测试。以下功能支持选择目标模型、设置初始种子和问题输入，并实时查看模糊测试结果。
    </p>

    <!-- 主表单 -->
    <el-form :model="form" label-width="120px">
      <!-- 模型选择 -->
      <el-form-item label="目标模型">
        <el-select
          v-model="form.model"
          placeholder="请选择模型"
          style="width: 300px"
        >
          <el-option label="GPT-3.5" value="gpt-3.5-turbo"></el-option>
          <el-option label="GPT-4" value="gpt-4o"></el-option>
          <el-option label="LLaMA2-7B" value="llama2-7b"></el-option>
          <el-option label="LLaMA3-8B" value="llama3-8b"></el-option>
          <el-option label="LLaMA3.1-8B" value="llama3.1-8b"></el-option>
        </el-select>
      </el-form-item>

      <!-- 问题输入 -->
      <el-form-item label="问题输入">
        <el-radio-group v-model="form.questionInputType">
          <el-radio label="default">使用默认问题</el-radio>
          <el-radio label="text">手动输入问题</el-radio>
        </el-radio-group>

        <!-- 统一的问题显示/输入区域 -->
        <div class="question-container">
          <!-- 默认问题选择和显示 -->
          <template v-if="form.questionInputType === 'default'">
            <el-select
              v-model="form.questionFile"
              placeholder="请选择问题文件"
              class="question-select"
              @change="handleQuestionFileChange"
            >
              <el-option
                v-for="file in questionFiles"
                :key="file"
                :label="file"
                :value="file"
              ></el-option>
            </el-select>
            <div class="question-content-area">
              <el-scrollbar>
                <div
                  v-for="(line, index) in selectedFileContent"
                  :key="index"
                  class="content-line"
                >
                  {{ line }}
                </div>
              </el-scrollbar>
            </div>
          </template>

          <!-- 手动输入问题 -->
          <template v-else>
            <div class="question-content-area">
              <el-input
                type="textarea"
                v-model="form.questionText"
                placeholder="请输入问题，每行一个"
                :rows="8"
              ></el-input>
            </div>
          </template>
        </div>
      </el-form-item>

      <!-- 测试设置 -->
      <!-- 任务描述 -->
      <el-form-item label="任务描述">
        <el-tooltip content="可选的任务描述" placement="right">
          <el-input
            v-model="form.task"
            placeholder=""
            type="text"
            style="width: 500px"
          ></el-input>
        </el-tooltip>
      </el-form-item>

      <el-form-item label="变异策略">
        <el-checkbox-group v-model="form.mutators">
          <el-checkbox label="Similar"></el-checkbox>
          <el-checkbox label="CrossOver"></el-checkbox>
          <el-checkbox label="Expand"></el-checkbox>
          <el-checkbox label="Rephrase"></el-checkbox>
          <el-checkbox label="Shorten"></el-checkbox>
          <el-checkbox label="TARGET_AWARE"></el-checkbox>
        </el-checkbox-group>
      </el-form-item>

      <el-form-item label="最大迭代次数">
        <el-input-number
          v-model="form.maxIterations"
          :min="1"
          :step="100"
          :controls="false"
          style="width: 100px"
        ></el-input-number>
      </el-form-item>

      <!-- 添加最大成功次数配置 -->
      <el-form-item label="最大成功次数">
        <el-tooltip
          content="每个问题达到此成功次数后将不再继续测试"
          placement="right"
        >
          <el-input-number
            v-model="form.maxSuccesses"
            :min="1"
            :max="10"
            :step="1"
            :controls="false"
            style="width: 100px"
          ></el-input-number>
        </el-tooltip>
      </el-form-item>

      <!-- 运行和下载按钮 -->
      <el-form-item>
        <el-button
          type="primary"
          :loading="isRunning"
          @click="handleRun"
          :disabled="isRunning"
        >
          {{ isRunning ? "测试进行中..." : "运行测试" }}
        </el-button>
        <el-button
          type="success"
          @click="handleDownload"
          :disabled="!canDownload"
          style="margin-left: 20px"
        >
          下载测试结果
        </el-button>
      </el-form-item>
    </el-form>

    <!-- 进度显示 -->
    <div class="progress-section">
      <el-progress
        :percentage="progress"
        :format="progressFormat"
        :status="progressStatus"
        :stroke-width="15"
      />
    </div>

    <!-- 日志输出区 -->
    <div class="log-section">
      <h3 class="log-title">日志输出</h3>
      <el-tabs v-model="activeLogTab" class="log-tabs">
        <el-tab-pane label="主要日志" name="main">
          <el-scrollbar ref="mainLogScrollbar" class="log-scrollbar">
            <div class="log-content">
              <div
                v-for="(log, index) in logs.main"
                :key="index"
                class="log-line"
              >
                {{ log }}
              </div>
            </div>
          </el-scrollbar>
        </el-tab-pane>
        <el-tab-pane label="变异日志" name="mutation">
          <el-scrollbar ref="mutationLogScrollbar" class="log-scrollbar">
            <div
              v-for="(log, index) in logs.mutation"
              :key="index"
              class="log-line"
            >
              {{ log }}
            </div>
          </el-scrollbar>
        </el-tab-pane>
        <el-tab-pane label="越狱日志" name="jailbreak">
          <el-scrollbar ref="jailbreakLogScrollbar" class="log-scrollbar">
            <div
              v-for="(log, index) in logs.jailbreak"
              :key="index"
              class="log-line"
            >
              {{ log }}
            </div>
          </el-scrollbar>
        </el-tab-pane>
        <el-tab-pane label="错误日志" name="error">
          <el-scrollbar ref="errorLogScrollbar" class="log-scrollbar">
            <div
              v-for="(log, index) in logs.error"
              :key="index"
              class="log-line error"
            >
              {{ log }}
            </div>
          </el-scrollbar>
        </el-tab-pane>
      </el-tabs>
    </div>

    <!-- 在日志部分后面添加 -->
    <div class="results-section">
      <h3>测试结果</h3>
      <el-tabs v-model="activeResultTab">
        <el-tab-pane label="实验总结" name="summary">
          <el-card class="result-card">
            <pre class="result-content">{{
              testResults?.experiment_summary || "暂无测试结果"
            }}</pre>
          </el-card>
        </el-tab-pane>
        <el-tab-pane label="详细结果" name="details">
          <el-card class="result-card">
            <div class="json-viewer">
              <pre class="result-content">{{ formattedAllResults }}</pre>
            </div>
          </el-card>
        </el-tab-pane>
        <el-tab-pane label="成功详情" name="success-details">
          <el-card class="result-card">
            <div
              v-if="testResults?.question_success_details"
              class="success-details"
            >
              <div
                v-for="(
                  details, question
                ) in testResults.question_success_details"
                :key="question"
                class="question-details"
              >
                <h4 class="question-title">{{ question }}</h4>
                <div v-if="details.length > 0">
                  <div
                    v-for="(result, index) in details"
                    :key="index"
                    class="success-item"
                  >
                    <div class="success-header">
                      <span class="iteration-label"
                        >迭代次数: {{ result.iteration }}</span
                      >
                    </div>
                    <div class="response-section">
                      <div class="response-item">
                        <div class="response-label">英文响应:</div>
                        <pre class="response-content">{{
                          result.response_en
                        }}</pre>
                      </div>
                      <div class="response-item">
                        <div class="response-label">中文响应:</div>
                        <pre class="response-content">{{
                          result.response_zh
                        }}</pre>
                      </div>
                    </div>
                  </div>
                </div>
                <div v-else class="no-success">该问题暂无成功案例</div>
              </div>
            </div>
            <div v-else class="no-data">暂无成功详情数据</div>
          </el-card>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ref,
  onMounted,
  onUnmounted,
  onBeforeUnmount,
  nextTick,
  computed,
} from "vue";
import { ElNotification, ElLoading } from "element-plus";
import {
  startFuzzing,
  stopFuzzing,
  getFuzzingStatus,
  getLogs,
  downloadResults,
  type FuzzingConfig,
  getQuestionFiles,
  getQuestionFileContent,
  clearSession as apiClearSession,
} from "../api/fuzzing";
import api from "../api";

const form = ref({
  model: "gpt-3.5-turbo",
  seedInputType: "default",
  questionInputType: "default",
  seedText: "",
  questionText: "",
  task: "",
  mutators: [],
  maxIterations: 1000,
  maxSuccesses: 3,
  questionFile: "",
});

const isRunning = ref(false);
const canDownload = ref(false);
const activeLogTab = ref("main");
const progress = ref(0);
const progressStatus = ref<"success" | "exception">("success");

// 日志状态
const logs = ref({
  main: [] as string[],
  mutation: [] as string[],
  jailbreak: [] as string[],
  error: [] as string[],
});

// 定时器
let statusInterval: number | null = null;
let logsInterval: number | null = null;

// 种子文件处理
const seedFileList = ref([]);
const beforeSeedUpload = (file) => {
  seedFileList.value = [file];
  return false; // 阻��默认上传行为
};
const handleSeedRemove = () => {
  seedFileList.value = [];
};

// 问题文件处理
const questionFileList = ref([]);
const beforeQuestionUpload = (file) => {
  questionFileList.value = [file];
  return false;
};
const handleQuestionRemove = () => {
  questionFileList.value = [];
};

const sessionId = ref("");
const questionFiles = ref<string[]>([]);
const selectedFileContent = ref<string[]>([]);

// 获取问题文件列表
const loadQuestionFiles = async () => {
  try {
    questionFiles.value = await getQuestionFiles();
  } catch (error) {
    ElNotification({
      title: "错误",
      message: "获���问题文件列表失败",
      type: "error",
    });
  }
};

// 处理问题文件选择
const handleQuestionFileChange = async (filename: string) => {
  try {
    const content = await getQuestionFileContent(filename);
    selectedFileContent.value = content;
  } catch (error) {
    ElNotification({
      title: "错误",
      message: "获取文件内容失败",
      type: "error",
    });
  }
};

// 运行测试
const handleRun = async () => {
  try {
    // 重置状态
    isRunning.value = true;
    logs.value = {
      main: [],
      mutation: [],
      jailbreak: [],
      error: [],
    };
    progress.value = 0;

    const config: FuzzingConfig = {
      model: form.value.model,
      seedInputType: form.value.seedInputType,
      questionInputType: form.value.questionInputType,
      seedContent: form.value.seedText,
      questionContent: form.value.questionText,
      questionFile: form.value.questionFile,
      mutators: form.value.mutators,
      maxIterations: form.value.maxIterations,
      maxSuccesses: form.value.maxSuccesses,
      task: form.value.task,
    };

    const response = await startFuzzing(config);
    sessionId.value = response.session_id;

    // 启动状态和日志轮询
    startPolling();

    ElNotification({
      title: "成功",
      message: "模糊测试已经开始",
      type: "success",
    });
  } catch (error) {
    isRunning.value = false;
    ElNotification({
      title: "错误",
      message: "启动测试失败: " + (error as Error).message,
      type: "error",
    });
  }
};

// 停止测试
const handleStop = async () => {
  try {
    await stopFuzzing();
    isRunning.value = false;
    stopPolling();

    ElNotification({
      title: "测试停止",
      message: "模糊测试停止",
      type: "warning",
    });
  } catch (error) {
    ElNotification({
      title: "错误",
      message: "模糊测试错误",
      type: "error",
    });
  }
};

// 开始轮询状态和日志
const startPolling = () => {
  if (!sessionId.value) return;

  statusInterval = window.setInterval(updateProgress, 1000);
  logsInterval = window.setInterval(async () => {
    if (!sessionId.value) return;

    try {
      const [mainLogs, mutationLogs, jailbreakLogs, errorLogs] =
        await Promise.all([
          getLogs(sessionId.value, "main", 0), // 获取全部日志
          getLogs(sessionId.value, "mutation", 0),
          getLogs(sessionId.value, "jailbreak", 0),
          getLogs(sessionId.value, "error", 0),
        ]);

      logs.value = {
        main: mainLogs.logs,
        mutation: mutationLogs.logs,
        jailbreak: jailbreakLogs.logs,
        error: errorLogs.logs,
      };

      // 在下一个tick更新滚动位置
      nextTick(() => {
        const scrollbars = {
          main: mainLogScrollbar.value,
          mutation: mutationLogScrollbar.value,
          jailbreak: jailbreakLogScrollbar.value,
          error: errorLogScrollbar.value,
        };

        // 获取当前激活��标签页对应的scrollbar
        const activeScrollbar = scrollbars[activeLogTab.value];
        if (activeScrollbar) {
          // 滚动到底部
          const wrap = activeScrollbar.$refs.wrap;
          if (wrap) {
            wrap.scrollTop = wrap.scrollHeight;
          }
        }
      });
    } catch (error) {
      console.error("Failed to get logs:", error);
    }
  }, 2000);
};

// 停止轮询
const stopPolling = () => {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
  if (logsInterval) {
    clearInterval(logsInterval);
    logsInterval = null;
  }
};

// 格式���进度
const progressFormat = (percentage: number) => {
  return `${Math.round(percentage)}%`;
};

// 添加清理会话的函数
const clearSession = async () => {
  try {
    await apiClearSession();
    // 重置所有状态
    isRunning.value = false;
    canDownload.value = false;
    sessionId.value = "";
    logs.value = {
      main: [],
      mutation: [],
      jailbreak: [],
      error: [],
    };
    progress.value = 0;
    // 重置结果为默认值
    testResults.value = {
      experiment_summary: "暂无测试结果",
      all_results: null,
    };
  } catch (error) {
    console.error("Failed to clear session:", error);
  }
};

// 修改 onMounted 钩子
onMounted(async () => {
  // 每次加载组件时先清理会话
  await clearSession();
  loadQuestionFiles();
});

// 修改 onBeforeUnmount 钩子
onBeforeUnmount(async () => {
  // 组件卸载前清理
  stopPolling();
  await clearSession();
});

// 添加ref用于获取scrollbar实例
const mainLogScrollbar = ref();
const mutationLogScrollbar = ref();
const jailbreakLogScrollbar = ref();
const errorLogScrollbar = ref();

// 添加新的响应式变量
const testResults = ref<any>({
  experiment_summary: "暂无测试结果",
  all_results: null,
});
const activeResultTab = ref("summary");

// 格式���所有结果的JSON
const formattedAllResults = computed(() => {
  if (!testResults.value?.all_results) return "暂无测试结果";
  try {
    if (typeof testResults.value.all_results === "string") {
      return testResults.value.all_results;
    }
    return JSON.stringify(testResults.value.all_results, null, 2);
  } catch (e) {
    console.error("Results formatting error:", e);
    return "结果格式化失败";
  }
});

// 修改handleDownload函数
const handleDownload = async () => {
  try {
    if (!sessionId.value) {
      ElNotification({
        title: "错误",
        message: "没有可用的测试结果",
        type: "error",
      });
      return;
    }

    // 添加加载状态
    const loading = ElLoading.service({
      lock: true,
      text: "正在加载测试结果...",
      background: "rgba(0, 0, 0, 0.7)",
    });

    try {
      const response = await downloadResults(sessionId.value);
      console.log("Download response:", response);

      // 确保数据存在且格式正确
      if (
        response &&
        (response.experiment_summary ||
          response.all_results ||
          response.question_success_details)
      ) {
        testResults.value = {
          experiment_summary: response.experiment_summary || "暂无实验总结",
          all_results: response.all_results || null,
          successful_jailbreaks: response.successful_jailbreaks || [],
          question_success_details: response.question_success_details || null,
        };

        ElNotification({
          title: "成功",
          message: "测试结果已加载",
          type: "success",
        });
      } else {
        ElNotification({
          title: "提示",
          message: "暂无测试结果数据",
          type: "warning",
        });
      }
    } finally {
      loading.close();
    }
  } catch (error) {
    console.error("Download error:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    ElNotification({
      title: "错误",
      message: "下载结果失败: " + errorMessage,
      type: "error",
    });
  }
};

// 修改进度条更新逻辑
const updateProgress = async () => {
  if (!sessionId.value) return;

  try {
    const status = await getFuzzingStatus(sessionId.value);
    if (status.is_running) {
      // 确保使用后端返回的实际迭代次数
      progress.value =
        status.current_iteration > 0
          ? (status.current_iteration / status.total_iterations) * 100
          : 0;

      // 添加进度显示
      progressStatus.value = "success";
    } else {
      stopPolling();
      isRunning.value = false;
      canDownload.value = true;

      if (status.current_iteration >= status.total_iterations) {
        progress.value = 100;
        progressStatus.value = "success";
      }

      // 自动加载结果
      await handleDownload();
    }
  } catch (error) {
    console.error("Failed to get status:", error);
    progressStatus.value = "exception";
  }
};
</script>

<style scoped>
.gpt-fuzzer-interface {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

h2 {
  margin-bottom: 10px;
}

p {
  margin-bottom: 20px;
}

.el-form-item {
  margin-bottom: 10px;
}

.log-section {
  margin-top: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 20px;
  height: 500px; /* 设置固定高度 */
  display: flex;
  flex-direction: column;
}

.log-title {
  margin-bottom: 15px;
  padding-left: 10px;
}

.log-tabs {
  margin-top: 10px;
}

.log-scrollbar {
  flex: 1; /* 让滚动区域占满剩余空间 */
  height: 400px;
  background: #f9f9f9;
  overflow-y: auto; /* 确保可以垂直滚动 */
}

.log-content {
  padding: 10px 15px;
  min-height: 100%; /* 确保内容区域至少和容器一样高 */
}

.log-line {
  font-family: monospace;
  padding: 4px 0;
  white-space: pre-wrap;
  line-height: 1.4;
  word-break: break-all; /* 防止长行破坏布局 */
}

.log-line.error {
  color: #f56c6c;
}

.progress-section {
  margin: 20px 0;
  padding: 10px;
  background: #fff;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

:deep(.el-tabs__content) {
  height: calc(100% - 40px); /* 减去tab标签的高度 */
  overflow: hidden;
}

:deep(.el-tab-pane) {
  height: 100%;
}

.question-container {
  margin-top: 10px;
  width: 100%;
}

.question-select {
  width: 300px;
  margin-bottom: 10px;
}

.question-content-area {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  height: 200px; /* 固定高度 */
  background-color: #fff;
  margin-top: 10px;
}

.question-content-area .el-scrollbar {
  height: 100%;
}

.question-content-area .el-textarea {
  height: 100%;
}

.question-content-area .el-textarea__inner {
  height: 100% !important;
  resize: none;
}

.content-line {
  padding: 8px 12px;
  border-bottom: 1px solid #f0f0f0;
}

.content-line:last-child {
  border-bottom: none;
}

/* 自定义滚动条样式 */
:deep(.el-scrollbar__wrap) {
  overflow-x: hidden !important;
}

:deep(.el-scrollbar__bar.is-horizontal) {
  display: none;
}

:deep(.el-scrollbar__bar.is-vertical) {
  width: 8px;
}

/* 添加结果显示相关样式 */
.results-section {
  margin-top: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 20px;
  background: #fff;
}

.result-card {
  margin-top: 10px;
  background: #f8f9fa;
}

.result-content {
  font-family: monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
  padding: 15px;
  margin: 0;
  font-size: 14px;
  line-height: 1.5;
  background: #f8f9fa;
  border-radius: 4px;
  color: #333;
}

.json-viewer {
  max-height: 500px;
  overflow-y: auto;
}

/* 美化JSON显示 */
.result-content .string {
  color: #008000;
}
.result-content .number {
  color: #0000ff;
}
.result-content .boolean {
  color: #b22222;
}
.result-content .null {
  color: #808080;
}
.result-content .key {
  color: #a52a2a;
}

/* 在已有的样式后面添加 */
.success-details {
  padding: 10px;
}

.question-details {
  margin-bottom: 24px;
  padding: 16px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  background-color: #fff;
}

.question-title {
  margin: 0 0 16px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid #ebeef5;
  color: #303133;
  font-size: 16px;
}

.success-item {
  margin-bottom: 16px;
  padding: 12px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.success-header {
  margin-bottom: 8px;
}

.iteration-label {
  font-weight: bold;
  color: #409eff;
}

.response-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.response-item {
  padding: 8px;
  background-color: #fff;
  border-radius: 4px;
}

.response-label {
  font-weight: bold;
  margin-bottom: 4px;
  color: #606266;
}

.response-content {
  margin: 0;
  padding: 8px;
  background-color: #f8f9fa;
  border-radius: 4px;
  font-family: monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-size: 14px;
  line-height: 1.5;
}

.no-success,
.no-data {
  text-align: center;
  color: #909399;
  padding: 20px;
}
</style>
