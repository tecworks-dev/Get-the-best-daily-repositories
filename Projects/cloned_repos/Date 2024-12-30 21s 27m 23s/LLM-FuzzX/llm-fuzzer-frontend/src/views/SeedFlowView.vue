<template>
  <div class="seed-flow-container">
    <div class="header-section">
      <h2>种子变异流可视化</h2>
      <div class="experiment-selector">
        <el-select
          v-model="selectedExperiment"
          placeholder="选择实验"
          @change="loadSeedFlow"
        >
          <el-option
            v-for="exp in experiments"
            :key="exp.id"
            :label="formatExperimentLabel(exp)"
            :value="exp.id"
            :disabled="!exp.has_seed_flow"
          >
            <span>{{ formatExperimentLabel(exp) }}</span>
            <span v-if="!exp.has_seed_flow" class="no-data-text">
              (无种子流数据)
            </span>
          </el-option>
        </el-select>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <el-spinner size="large" />
      <p>正在加载种子流数据...</p>
    </div>

    <!-- 错误提示 -->
    <div v-if="error" class="error-container">
      <el-alert :title="error" type="error" show-icon />
    </div>

    <!-- 种子流可视化区域 -->
    <div class="visualization-container" ref="visualizationContainer">
      <svg ref="svgContainer"></svg>
    </div>

    <!-- 详细信息弹出框 -->
    <div
      v-if="hoveredNode"
      class="node-details"
      :style="nodeDetailsStyle"
      @mouseleave="hoveredNode = null"
    >
      <div class="details-header">
        <h3>种子详情</h3>
        <span class="details-id">ID: {{ hoveredNode.id }}</span>
      </div>
      <div class="details-content">
        <div class="details-section">
          <h4>基本信息</h4>
          <p>
            <strong>变异类型:</strong> {{ hoveredNode.mutation_type || "Root" }}
          </p>
          <p><strong>深度:</strong> {{ hoveredNode.depth }}</p>
        </div>
        <div class="details-section">
          <h4>统计信息</h4>
          <p><strong>使用次数:</strong> {{ hoveredNode.stats.uses }}</p>
          <p><strong>成功次数:</strong> {{ hoveredNode.stats.successes }}</p>
          <p>
            <strong>总尝试次数:</strong> {{ hoveredNode.stats.total_trials }}
          </p>
          <p>
            <strong>成功率:</strong>
            {{ calculateSuccessRate(hoveredNode.stats) }}%
          </p>
        </div>
        <div class="details-section">
          <h4>内容</h4>
          <div class="content-box">
            <pre>{{ hoveredNode.content }}</pre>
          </div>
        </div>
      </div>
    </div>

    <!-- 添加日志输出区 -->
    <div class="log-section">
      <h3 class="log-title">实验日志</h3>
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

    <!-- 添加结果显示区域 -->
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
import { ref, onMounted, watch, nextTick, computed } from "vue";
import * as d3 from "d3";
import {
  getExperiments,
  getSeedFlow,
  getExperimentLogs,
  getExperimentResults,
} from "../api/fuzzing";
import type {
  Experiment,
  SeedFlowData,
  ExperimentLogs,
  ExperimentResults,
} from "../api/fuzzing";

// 状态变量
const experiments = ref<Experiment[]>([]);
const selectedExperiment = ref<string>("");
const loading = ref(false);
const error = ref<string | null>(null);
const hoveredNode = ref<any>(null);
const nodeDetailsStyle = ref({
  top: "0px",
  left: "0px",
});

// DOM引用
const visualizationContainer = ref<HTMLDivElement | null>(null);
const svgContainer = ref<SVGElement | null>(null);

// 格式化实验标签
const formatExperimentLabel = (exp: Experiment) => {
  const date = new Date(exp.timestamp);
  return `实验 ${exp.id} (${date.toLocaleString()})`;
};

// 计算成功率
const calculateSuccessRate = (stats: any) => {
  if (!stats || !stats.total_trials) return 0;
  return ((stats.successes / stats.total_trials) * 100).toFixed(2);
};

// 格式化日期
const formatDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleString();
};

// 加载实验列表
const loadExperiments = async () => {
  try {
    experiments.value = await getExperiments();
  } catch (e) {
    error.value = "加载实验列表失败";
    console.error(e);
  }
};

// 处理种子流数据
const processSeedFlowData = (data: SeedFlowData) => {
  // 构建树形结构
  const nodesMap = new Map();

  // 首先创建所有节点
  Object.entries(data).forEach(([id, nodeData]) => {
    nodesMap.set(id, {
      id: id,
      data: nodeData,
      children: [],
    });
  });

  // 建立父子关系并找出所有根节点
  const roots = [];
  nodesMap.forEach((node, id) => {
    const parentId = data[id].parent_id;
    if (parentId && nodesMap.has(parentId)) {
      const parent = nodesMap.get(parentId);
      parent.children.push(node);
    } else {
      // 如果没有父节点或父节点不存在，就作为根节点
      roots.push(node);
    }
  });

  // 返回森林结构
  return {
    id: "virtual_root",
    children: roots,
    data: {
      content: "Root",
      mutation_type: "root",
      stats: { uses: 0, successes: 0, total_trials: 0 },
      depth: -1,
    },
  };
};

// 绘制树形图
const drawTree = (data: any) => {
  if (!visualizationContainer.value || !svgContainer.value) return;

  // 清除现有内容
  d3.select(svgContainer.value).selectAll("*").remove();

  // 设置尺寸和边距
  const margin = { top: 50, right: 30, bottom: 50, left: 30 };
  const width =
    visualizationContainer.value.clientWidth - margin.left - margin.right;

  // 创建层次结构
  const root = d3.hierarchy(data);

  // 计算树的高度，减小节点垂直间距
  const nodeHeight = 20; // 节点垂直间距
  const height = Math.max(600, (root.height + 2) * nodeHeight); // 将最小高度从 500 改为 800

  // 创建SVG
  const svg = d3
    .select(svgContainer.value)
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom);

  const g = svg
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // 创建树形布局
  const treeLayout = d3
    .tree()
    .size([width, height])
    .separation((a, b) => {
      // 调整相邻节点之间的水平间距
      return a.parent === b.parent ? 1.2 : 1.5; // 将原来的 2 改为 1.5，减小水平间距
    });

  // 应用布局
  const treeData = treeLayout(root);

  // 绘制连接线
  const links = g
    .selectAll(".link")
    .data(treeData.links())
    .enter()
    .append("path")
    .attr("class", "link")
    .attr(
      "d",
      d3
        .linkVertical()
        .x((d: any) => d.x)
        .y((d: any) => d.y)
    )
    .style("fill", "none")
    .style("stroke", "#e9ecef")
    .style("stroke-width", "1.5px");

  // 创建节点组
  const nodes = g
    .selectAll(".node")
    .data(treeData.descendants())
    .enter()
    .append("g")
    .attr(
      "class",
      (d) => `node ${d.children ? "node--internal" : "node--leaf"}`
    )
    .attr("transform", (d: any) => `translate(${d.x},${d.y})`);

  // 添加节点圆圈
  nodes
    .append("circle")
    .attr("r", 8)
    .style("fill", (d: any) => {
      // 跳过虚拟根节点
      if (d.data.id === "virtual_root") return "none";

      const stats = d.data.data.stats;
      const successRate = calculateSuccessRate(stats);
      // 使用更柔和的配色方案
      if (successRate === 0) return "#e0e0e0"; // 灰色表示未使用
      if (successRate < 30) return "#74c0fc"; // 浅蓝色表示低成功率
      if (successRate < 70) return "#4dabf7"; // 中等蓝色
      return "#228be6"; // 深蓝色表示高成功率
    })
    .style("stroke", "#666")
    .style("stroke-width", "1.5px")
    .style("display", (d) => (d.data.id === "virtual_root" ? "none" : null))
    .on("mouseover", (event: MouseEvent, d: any) => {
      // 跳过虚拟根节点
      if (d.data.id === "virtual_root") return;

      hoveredNode.value = d.data.data;
      const rect = (event.target as Element).getBoundingClientRect();

      // 计算悬浮框位置，避免超出视窗
      const detailsWidth = 300;
      const detailsHeight = 400;

      // 计算初始位置
      let left = rect.right + 10;
      let top = rect.top + window.scrollY;

      // 确保不超出右边界
      if (left + detailsWidth > window.innerWidth) {
        left = rect.left - detailsWidth - 10;
      }

      // 确保不超出底部，如果会超出则显示在上方
      if (top + detailsHeight > window.innerHeight + window.scrollY) {
        if (rect.top - detailsHeight > 0) {
          // 如果上方空间足够，显示在上方
          top = rect.top - detailsHeight + window.scrollY;
        } else {
          // 如果上方空间不够，固定在视窗底部
          top = window.innerHeight + window.scrollY - detailsHeight - 10;
        }
      }

      nodeDetailsStyle.value = {
        position: "absolute",
        top: `${top}px`,
        left: `${left}px`,
        maxHeight: `${detailsHeight}px`,
        overflow: "auto",
      };
    });

  // 添加节点标签
  /*
  nodes
    .append("text")
    .attr("dy", (d: any) => (d.children ? -13 : 25))
    .attr("x", 0)
    .style("text-anchor", "middle")
    .style("display", (d) => (d.data.id === "virtual_root" ? "none" : null))
    .text((d: any) => {
      if (d.data.id === "virtual_root") return "";
      // 只显示ID的最后6位作为标签
      return d.data.id.slice(-6);
    })
    .style("font-size", "12px")
    .style("font-family", "Arial");
  */
};

// 添加新的响应式变量
const activeLogTab = ref("main");
const activeResultTab = ref("summary");
const logs = ref<ExperimentLogs>({
  main: [],
  mutation: [],
  jailbreak: [],
  error: [],
});
const testResults = ref<ExperimentResults>({
  experiment_summary: "暂无测试结果",
  all_results: null,
  question_success_details: {},
});

// 格式化所有结果的JSON
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

// 加载种子流数据
const loadSeedFlow = async (experimentId?: string) => {
  if (!experimentId && !selectedExperiment.value) return;

  loading.value = true;
  error.value = null;

  try {
    const currentId = experimentId || selectedExperiment.value;

    // 并行加载所有数据
    const [seedFlowData, logData, resultsData] = await Promise.all([
      getSeedFlow(currentId),
      getExperimentLogs(currentId),
      getExperimentResults(currentId),
    ]);

    // 处理种子流数据
    const processedData = processSeedFlowData(seedFlowData);
    await nextTick();
    drawTree(processedData);

    // 更新日志
    logs.value = logData;

    // 更新测试结果
    testResults.value = resultsData;
  } catch (e) {
    error.value = "加载数据失败";
    console.error(e);
  } finally {
    loading.value = false;
  }
};

// 监听窗口大小变化
const handleResize = () => {
  if (selectedExperiment.value) {
    loadSeedFlow(selectedExperiment.value);
  }
};

// 生命周期钩子
onMounted(async () => {
  await loadExperiments();
  window.addEventListener("resize", handleResize);
});

watch(selectedExperiment, (newValue) => {
  if (newValue) {
    loadSeedFlow(newValue);
  }
});
</script>

<style scoped>
.seed-flow-container {
  padding: 20px;
  height: calc(100vh - 100px);
  display: flex;
  flex-direction: column;
}

.header-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.experiment-selector {
  width: 300px;
}

.no-data-text {
  color: #999;
  margin-left: 10px;
}

.visualization-container {
  flex: 1;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  overflow: auto;
  padding: 20px;
  min-height: 600px;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
}

.error-container {
  margin: 20px 0;
}

.node-details {
  position: absolute;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  padding: 16px;
  width: 300px;
  max-height: 400px;
  overflow-y: auto;
  z-index: 1000;
}

.details-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid #eee;
}

.details-header h3 {
  margin: 0;
  color: #333;
}

.details-id {
  color: #666;
  font-size: 0.9em;
}

.details-section {
  margin-bottom: 16px;
}

.details-section h4 {
  color: #666;
  margin: 8px 0;
}

.content-box {
  background: #f8f9fa;
  border-radius: 4px;
  padding: 12px;
  max-height: 150px; /* 限制内容框高度 */
  overflow-y: auto;
}

.content-box pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: monospace;
  font-size: 12px;
}

/* 添加滚动条样式 */
.content-box::-webkit-scrollbar {
  width: 6px;
}

.content-box::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.content-box::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}

.content-box::-webkit-scrollbar-thumb:hover {
  background: #555;
}

:deep(.link) {
  fill: none;
  stroke: #e9ecef; /* 更浅连接线颜色 */
  stroke-width: 1.5px;
}

:deep(.node circle) {
  cursor: pointer;
  transition: r 0.2s;
}

:deep(.node circle:hover) {
  r: 10;
}

:deep(.node text) {
  font-size: 12px;
  font-family: Arial;
}

/* 节点样式 */
:deep(.node--internal circle) {
  stroke-width: 2px;
}

:deep(.node--leaf circle) {
  stroke-width: 1.5px;
}

/* 添加滚动条样式 */
.node-details::-webkit-scrollbar {
  width: 6px;
}

.node-details::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.node-details::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}

.node-details::-webkit-scrollbar-thumb:hover {
  background: #555;
}

/* 添加日志和结果相关样式 */
.log-section {
  margin-top: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 20px;
  height: 700px;
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
  flex: 1;
  height: 700px;
  background: #f9f9f9;
  overflow-y: auto;
}

.log-content {
  padding: 10px 15px;
  min-height: 100%;
}

.log-line {
  font-family: monospace;
  padding: 4px 0;
  white-space: pre-wrap;
  line-height: 1.4;
  word-break: break-all;
}

.log-line.error {
  color: #f56c6c;
}

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
