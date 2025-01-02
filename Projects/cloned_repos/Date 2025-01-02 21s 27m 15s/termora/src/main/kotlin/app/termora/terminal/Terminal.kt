package app.termora.terminal

/**
 * ALL METHODS ARE NOT THREAD SAFE
 *
 * 终端的所有操作都不是线程安全的！
 */
interface Terminal {

    /**
     * 写入内容到终端
     */
    fun write(text: String)

    /**
     * 文档
     */
    fun getDocument(): Document

    /**
     * 搜索服务
     */
    fun getFindModel(): FindModel

    /**
     * 终端
     */
    fun getTerminalModel(): TerminalModel

    /**
     * 选择
     */
    fun getSelectionModel(): SelectionModel

    /**
     * 光标
     */
    fun getCursorModel(): CursorModel

    /**
     * 标记
     */
    fun getMarkupModel(): MarkupModel

    /**
     * 滚动条
     */
    fun getScrollingModel(): ScrollingModel

    /**
     * 键盘编码
     */
    fun getKeyEncoder(): KeyEncoder

    /**
     * Tab
     */
    fun getTabulator(): Tabulator

    /**
     * 关闭
     */
    fun close()

    /**
     * 添加监听器
     */
    fun addTerminalListener(listener: TerminalListener)

    /**
     * 获取监听器列表
     */
    fun getTerminalListeners(): List<TerminalListener>

    /**
     * 移除监听器
     */
    fun removeTerminalListener(listener: TerminalListener)

}

