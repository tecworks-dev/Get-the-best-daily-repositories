import { ChartingLibraryFeatureset } from "@/public/charting_library/charting_library";

// 定义要禁用的功能列表
export const disabled_features: ChartingLibraryFeatureset[] = [
  "symbol_search_hot_key", // 禁用快捷键搜索交易对的功能
  "show_object_tree", // 禁用对象树功能（用于管理图表上的绘制对象）
  "object_tree_legend_mode", // 禁用对象树中的图例模式
  "popup_hints", // 禁用图表中的弹出提示信息
  // "legend_widget", // 控制是否显示图表图例（显示商品数据和信息）
  "symbol_info", // 禁用商品信息对话框（显示当前交易对的详细信息）
  // "header_undo_redo", // 禁用头部的撤销/重做按钮
  "header_compare", // 禁用头部的“比较”按钮（用于比较多个商品的价格走势）
  // "header_saveload", // 禁用头部的保存/加载功能
  // "header_screenshot", // 控制是否显示截图按钮（用于捕捉图表）
  "timeframes_toolbar", // 禁用时间框工具栏（用于切换不同时间周期的快捷方式）
  /// "header_fullscreen_button", // 控制是否显示全屏按钮
  // "use_localstorage_for_settings", // 允许将用户设置保存到 localStorage
  // "save_chart_properties_to_local_storage", // 禁用保存图表属性到 localStorage 的功能
  "header_symbol_search", // 禁用头部的搜索框（用于搜索交易对）
  "volume_force_overlay", // 禁用强制将交易量与价格分开显示的功能
  "datasource_copypaste", // 禁用数据源的复制粘贴功能
  "main_series_scale_menu", // 隐藏图表右下角的设置菜单
  "display_market_status", // 禁用市场状态显示（如“市场关闭”等）
  // "dont_show_boolean_study_arguments", // 隐藏布尔型指标的参数（true/false）
  // "hide_last_na_study_output", // 隐藏最后的 N/A 指标输出数据
  // "hide_resolution_in_legend", // 在图表图例和数据窗口中隐藏时间周期（如 D、W 等）
  // "left_toolbar", // 控制是否隐藏左侧工具栏
  // "header_widget", // 隐藏图表头部区域
  // "header_settings", // 禁用头部设置按钮
  // "header_chart_type", // 禁用头部图表类型选择功能
  // "header_screenshot", // 禁用头部截图按钮
  "source_selection_markers", // 禁用来源选择标记功能
];
export const enabled_features: ChartingLibraryFeatureset[] = [
  "seconds_resolution", // 支持秒级的时间周期，如果禁用，则无法查看秒级别的 K 线图
  "two_character_bar_marks_labels", // 禁用两字符的 K 线标记标签
  // "dont_show_boolean_study_arguments", // 隐藏布尔型（true/false）指标参数
  // "hide_last_na_study_output", // 隐藏最后的 N/A 指标输出数据
  // "hide_resolution_in_legend", // 在图例和数据窗口中隐藏时间周期（如 D、W 等）
  // "left_toolbar", // 隐藏图表左侧工具栏
  // "header_widget", // 隐藏头部区域（包括功能按钮）
  // "show_dom_first_time", // 禁用首次显示 DOM 数据
  // "fix_left_edge", // 阻止用户滚动到第一个历史 K 线的左侧
  // "hide_unresolved_symbols_in_legend", // 在图例中隐藏未解析的指标参数
  // "horz_touch_drag_scroll", // 禁用水平触摸拖动滚动功能
  // "right_bar_stays_on_scroll", // 滚动时右侧工具栏保持可见
  // "lock_visible_time_range_on_resize", // 防止在调整图表大小时更改可见的时间范围
  // "cropped_tick_marks", // 如果禁用，则部分可见的价格标签在价格轴上会被隐藏
  // "display_market_status", // 禁用市场状态显示（如市场是否关闭的信息）
  // "countdown", // 禁用倒计时功能（用于显示下一根 K 线的倒计时）
];
