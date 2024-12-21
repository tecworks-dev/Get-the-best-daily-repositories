/** @type {import('tailwindcss').Config} */
const defaultTheme = require("tailwindcss/defaultTheme");
const colors = require("tailwindcss/colors");
module.exports = {
  content: [
    "./app.html",
    "./components/**/*.{js,vue,ts}",
    "./layouts/**/*.vue",
    "./pages/**/*.vue",
    "./plugins/**/*.{js,ts}",
    "./nuxt.config.{js,ts}",
    "./locales/*.json",
  ],
  theme: {
    extend: {
      // Set theme colors (Required config!)
      colors: {
        primary: colors.blue,
        secondary: colors.slate,
      },
      keyframes: {
        circle: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
      },
      animation: {
        circle: "circle 1s ease",
      },
    },
  },
  // Add plugins
  plugins: [require("@tailwindcss/typography"), require("@tailwindcss/forms")],
};
