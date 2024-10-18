/** @type {import('@lingui/conf').LinguiConfig} */
const localeNames = require("./framework/locale/localeConfig")
//https://www.npmjs.com/package/@lingui/format-json
const formatter = require("@lingui/format-json")
console.info("Using LinguiJS config",formatter)
module.exports = {
    // locales: ["zh-CN","en", "cs", "fr","de","es","it","ja","ko","nl","pt-BR","ru","uk","vi","zh-TW","pt","id"],
    locales: Object.keys(localeNames),
    sourceLocale: "en",
    format: formatter.formatter({style: "lingui"}),
    formatOptions: {
        lineNumbers: false,
    },
    catalogs: [
        {
            path: "<rootDir>/translations/{locale}/messages",
            include: ["app", "components", "lib", "config", "utils","framework"],
            exclude: ["**/.next/**", "**/*.d.ts", "**/node_modules/**"],
        },
    ],
};