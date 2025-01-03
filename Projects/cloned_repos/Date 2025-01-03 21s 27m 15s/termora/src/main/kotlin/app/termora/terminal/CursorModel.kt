package app.termora.terminal

enum class CursorStyle {
    Block,
    Bar,
    Underline
}

enum class CursorMove {
    /**
     * 向左移动一次
     */
    Left,

    /**
     * 向右移动一次
     */
    Right,

    /**
     * 向上移动一次
     */
    Up,

    /**
     * 向下移动一次
     */
    Down,

    /**
     * 回到行尾
     */
    RowEnd,

    /**
     * 回到行首
     */
    RowHome,
}

interface CursorModel {

    /**
     * 获取终端信息
     */
    fun getTerminal(): Terminal

    /**
     * 获取光标位置
     */
    fun getPosition(): Position

    /**
     * 全屏模式下的光标
     */
    fun getAlternateScreenBufferCursorModel(): CursorModel

    /**
     * 非全屏模式下的光标
     */
    fun getNonAlternateScreenBufferCursorModel(): CursorModel

    /**
     * 移动光标
     */
    fun move(move: CursorMove)

    /**
     * 移动光标 N 次
     */
    fun move(move: CursorMove, count: Int)

    /**
     * 移动光标
     */
    fun move(row: Int, col: Int)

    /**
     * 获取光标样式
     */
    fun getStyle(): CursorStyle
}