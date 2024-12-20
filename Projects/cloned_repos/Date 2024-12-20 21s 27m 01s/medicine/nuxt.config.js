const TerserPlugin = require("terser-webpack-plugin");
const tableData = require("./data/tableData.json");

export default {
  target: "static",
  loading: false,
  // Global page headers: https://go.nuxtjs.dev/config-head
  head() {
    return {
      title: "进口原研药目录",
      meta: [
        { charset: "utf-8" },
        { name: "viewport", content: "width=device-width, initial-scale=1" },
        {
          hid: "description",
          name: "description",
          content:
            "约 400 种药品，实时更新的进口原研药数据库",
        },
      ],
      link: [{ rel: "icon", type: "image/x-icon", href: "/favicon.ico" }],
    };
  },

  // Global CSS: https://go.nuxtjs.dev/config-css
  css: ["~/assets/main.css"],

  // Plugins to run before rendering page: https://go.nuxtjs.dev/config-plugins
  plugins: [ "~/plugins/vue-good-table.js"],

  // Auto import components: https://go.nuxtjs.dev/config-components
  components: true,

  // Modules for dev and build (recommended): https://go.nuxtjs.dev/config-modules
  buildModules: [
    // https://go.nuxtjs.dev/tailwindcss
    // "@nuxtjs/tailwindcss",
  ],

  // Modules: https://go.nuxtjs.dev/config-modules
  modules: ["@nuxtjs/axios", "@nuxtjs/i18n", "@nuxtjs/sitemap", "vue-social-sharing/nuxt"],


  server: {
    port: 8000, // default: 3000
  },

  // sitemap
  sitemap: {
    hostname: "https://medicine.lvwzhen.com",
    gzip: true,
    trailingSlash: true,
  },

  // Build Configuration: https://go.nuxtjs.dev/config-build
  build: {
    extractCSS: true,
    postcss: {
      postcssOptions: {
        plugins: {
          tailwindcss: {},
          autoprefixer: {},
        },
      },
    },
    optimization: {
      minimizer: [
        new TerserPlugin({
          parallel: 4,
          terserOptions: {
            compress: {
              warnings: true,
              drop_debugger: true, // 删除 debugger
              drop_console: true, // 删除 console
            },
          },
        }),
      ],
    },
  },
  // generate
  generate: {
    fallback: "404.html",
  },
};
