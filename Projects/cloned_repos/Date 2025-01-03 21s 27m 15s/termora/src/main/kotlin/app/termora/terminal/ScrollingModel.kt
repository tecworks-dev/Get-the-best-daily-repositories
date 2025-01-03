package app.termora.terminal

interface ScrollingModel {

    companion object {
        /**
         * 滚动事件
         */
        val Scroll = DataKey(Int::class)
    }

    fun getTerminal(): Terminal

    /**
     * 获取垂直偏移
     *
     * @return 偏移列数
     */
    fun getVerticalScrollOffset(): Int

    /**
     * 获取允许的最大偏移，总行数 - rows = 最大偏移
     *
     * @return 当返回的数字小于[Document.getLines]的数量时，表示不可滚动
     */
    fun getMaxVerticalScrollOffset(): Int

    /**
     * 全屏模式下的滚动
     */
    fun getAlternateScreenBufferScrollingModel(): ScrollingModel

    /**
     * 非全屏模式下的滚动
     */
    fun getNonAlternateScreenBufferScrollingModel(): ScrollingModel

    /**
     * 是否粘附在最底部。如果返回 true 那么会自动滚动到最底部。
     */
    fun isStick(): Boolean

    /**
     * 是否允许垂直滚动，当行数不足时会反悔 false 那么就不准显示垂直滚动条
     */
    fun canVerticalScroll(): Boolean

    /**
     * 滚动到指定列
     */
    fun scrollTo(offset: Int)

    /**
     * 滚动到指定行
     */
    fun scrollToRow(row: Int)

    /**
     * 判断此行是否在可视区域内
     */
    fun isInVisibleArea(row: Int): Boolean


}