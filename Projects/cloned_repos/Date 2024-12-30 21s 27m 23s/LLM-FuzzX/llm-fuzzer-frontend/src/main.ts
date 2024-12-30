import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import store from "./store";

// 引入 Element Plus
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";

createApp(App).use(router).use(store).use(ElementPlus).mount("#app");
