import { RouteRecordRaw } from "vue-router";
import HomeView from "../views/HomeView.vue";
import SeedFlowView from "../views/SeedFlowView.vue";

const routes: Array<RouteRecordRaw> = [
  {
    path: "/",
    name: "Home",
    component: HomeView,
  },
  {
    path: "/seed-flow",
    name: "SeedFlow",
    component: SeedFlowView,
  },
];

export default routes;
