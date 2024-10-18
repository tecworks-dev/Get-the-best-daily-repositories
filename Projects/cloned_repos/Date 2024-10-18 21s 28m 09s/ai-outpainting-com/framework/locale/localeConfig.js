// 定义当前项目所需要国际化的语言包，所有国际化配置均依次配置实现
// 当前文件不可改为ts文件。在scripts/translate-locale.js文件中会使用到当前配置文件。
// 只可以使用module.exports导出一个对象，对象中包含所有语言的名称和对应的语言代码。
// 新增或者删除语言时，需要同时修改./locale.ts#AVAILABLE_LOCALES的值
module.exports = {
   "en": 'English', // 英语
   "cs": 'Čeština', // 捷克语
   "fr": 'Français', // 法语
   "de": 'Deutsch', // 德语
   "es": 'Español', // 西班牙语
   "it": 'Italiano', // 意大利语
   "ja": '日本語', // 日语
   "ko": '한국어', // 韩语
   "nl": 'Nederlands', // 荷兰语
   "pt-BR": 'Português do Brasil', // 巴西葡萄牙语
   "ru": 'Русский', // 俄语
   "uk": 'Українська', // 乌克兰语
   "vi": 'Tiếng Việt', // 越南语
   "zh-TW": '繁体中文', // 繁体中文
   "pt": 'Português', // 葡萄牙语
   "da": 'Dansk', // 丹麦语
   "el": 'Ελληνικά (Elliniká)', // 希腊语
   "no": 'Norsk', // 挪威语
   "fi": 'Suomi', // 芬兰语
   "sv": 'Svenska', // 瑞典语
   "th": 'ไทย (Thai)', // 泰语
   "id": 'Bahasa Indonesia', // 印度尼西亚语
   "hi": 'हिन्दी (Hindi)', // 印地语
   // "ar": 'العربية (Arabic)', // 阿拉伯语 （从右往左）
   "bn": 'বাংলা (Bangla)', // 孟加拉语
   "ms": 'Bahasa Melayu', // 马来语
   "tr": 'Türkçe', // 土耳其语
   // "fa": 'فارسی (Farsi)', // 波斯语 （从右往左）
}
