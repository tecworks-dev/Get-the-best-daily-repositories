<template>
  <header class="global-header">
    <div class="logo-container">
      <img src="../assets/logo.svg" alt="Logo" class="logo" />
      <span class="app-name">FuzzLLM</span>
    </div>
    <nav>
      <el-menu mode="horizontal" :default-active="$route.path" router>
        <el-menu-item index="/">首页</el-menu-item>
        <el-menu-item index="/seed-flow">种子流可视化</el-menu-item>
      </el-menu>
    </nav>
    <div class="user-greeting">
      Hello, admin1024
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();

const filteredRoutes = computed(() => {
  return router.options.routes.filter((route) => route.name);
});
</script>

<style scoped>
.global-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 40px;
  background-color: #ffffff;
  border-bottom: 1px solid #e0e0e0;
  height: 60px;
  margin-bottom: 0;
  margin: 0; /* 确保没有底部间隙 */
}

.logo-container {
  display: flex;
  align-items: center;
}

.logo {
  height: 40px;
  margin-right: 10px;
}

.app-name {
  font-size: 20px;
  font-weight: bold;
  color: #333;
}

nav {
  flex-grow: 1;
  margin-left: 50px;
}

.el-menu {
  background-color: transparent;
  border-bottom: none;
}

.el-menu-item {
  color: #333;
  font-size: 16px;
}

.el-menu-item.is-active {
  color: #409eff;
}

.user-greeting {
  /* 新增的样式 */
  margin-left: auto; /* 确保文本在最右侧 */
  font-size: 16px;
  color: #333;
}
</style>
