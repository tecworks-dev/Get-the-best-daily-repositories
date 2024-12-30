<template>
  <BasicLayout>
    <router-view />
  </BasicLayout>
</template>

<script setup lang="ts">
// 引入基本布局组件
import BasicLayout from "./layouts/BasicLayout.vue";
</script>

<style scoped>
/* 在这里可以添加全局的样式 */
body {
  font-family: "Roboto", sans-serif;
  margin: 0;
  padding: 0;
  background-color: #f9f9f9;
}
</style>
