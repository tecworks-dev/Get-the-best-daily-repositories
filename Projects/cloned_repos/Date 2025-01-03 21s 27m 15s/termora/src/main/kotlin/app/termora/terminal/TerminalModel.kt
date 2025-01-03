package app.termora.terminal

interface DataListener {
    fun onChanged(key: DataKey<*>, data: Any) {}
}

/**
 * Terminal 事件
 */
interface TerminalAction {
    /**
     * 是否匹配
     */
    fun matches(e: TerminalEvent): Boolean

    /**
     * 触发
     */
    fun actionPerformed(e: TerminalEvent)

    /**
     * 名称
     */
    fun name(): String {
        return this.toString()
    }
}

data class TerminalSize(val rows: Int, val cols: Int)
data class TerminalResize(val oldSize: TerminalSize, val newSize: TerminalSize)
open class TerminalKeyEvent(val keyCode: Int, modifiers: Int = 0) :
    TerminalEvent(modifiers) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true

        other as TerminalKeyEvent

        if (keyCode != other.keyCode) return false
        if (modifiers != other.modifiers) return false

        return true
    }

    override fun hashCode(): Int {
        var result = keyCode
        result = 31 * result + modifiers
        return result
    }

    override fun toString(): String {
        return "keyCode: $keyCode, modifiers: $modifiers"
    }
}

interface TerminalModel {
    companion object {
        /**
         * 当 terminal 调用 [resize] 的时候会发出 [DataListener.onChanged] 事件
         */
        val Resize = DataKey(TerminalResize::class)
    }

    fun getTerminal(): Terminal

    /**
     * 获取列数
     */
    fun getCols(): Int

    /**
     * 获取行数
     */
    fun getRows(): Int

    /**
     * 允许的最大行数，超出则会被清除
     */
    fun getMaxRows(): Int

    /**
     * 获取颜色版
     */
    fun getColorPalette(): ColorPalette

    /**
     * 获取数据或开关
     */
    fun <T : Any> getData(key: DataKey<T>): T

    /**
     * 获取数据或开关
     */
    fun <T : Any> getData(key: DataKey<T>, defaultValue: T): T

    /**
     * 设置数据或开关
     */
    fun <T : Any> setData(key: DataKey<T>, data: T)


    /**
     * 是否包含
     */
    fun hasData(key: DataKey<*>): Boolean


    /**
     * 添加监听器
     */
    fun addDataListener(listener: DataListener)

    /**
     * 删除监听器
     */
    fun removeDataListener(listener: DataListener)

    /**
     * Bell
     */
    fun bell()

    /**
     * 修改大小
     */
    fun resize(rows: Int, cols: Int)

}

/**
 * 是否是全屏模式
 */
fun TerminalModel.isAlternateScreenBuffer(): Boolean {
    return getData(DataKey.AlternateScreenBuffer, false)
}

/**
 * 返回滚动区域
 */
fun TerminalModel.getScrollingRegion(): ScrollingRegion {
    return getData(DataKey.ScrollingRegion)
}

/**
 * 光标起源模式
 */
fun TerminalModel.isOriginMode(): Boolean {
    return getData(DataKey.OriginMode, false)
}

