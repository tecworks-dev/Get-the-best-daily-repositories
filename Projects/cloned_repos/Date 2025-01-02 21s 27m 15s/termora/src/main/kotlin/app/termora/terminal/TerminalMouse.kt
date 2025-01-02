package app.termora.terminal

// X11 button number
enum class TerminalMouseButton(val code: Int) {
    /**
     * no button
     */
    None(-1),

    /**
     * left button
     */
    Left(0),

    /**
     * middle button
     */
    Middle(1),

    /**
     * right button
     */
    Right(2),

    /**
     * scroll down
     */
    ScrollDown(4),

    /**
     * scroll up
     */
    ScrollUp(5),

}

enum class TerminalMouseEventType {
    Pressed, Released, Moved,
}


/**
 * @param button 鼠标键
 * @param count 点击次数
 */
open class TerminalMouseEvent(val button: TerminalMouseButton, val count: Int = 1, modifiers: Int) :
    TerminalEvent(modifiers)


/**
 * 鼠标
 */
interface TerminalMouse {
    /**
     * 鼠标点击事件。 x y 是像素点
     */
    fun mousePressed(x: Int, y: Int, event: TerminalMouseEvent) {}


    /**
     * 鼠标点击事件。 x y 是像素点
     */
    fun mouseClicked(x: Int, y: Int, event: TerminalMouseEvent) {}

    /**
     * 鼠标释放。 x y 是像素点
     */
    fun mouseReleased(x: Int, y: Int, event: TerminalMouseEvent) {}

    /**
     * 鼠标滚动
     */
    fun mouseWheel(button: TerminalMouseButton, event: TerminalMouseEvent) {}

    /**
     * 鼠标移动
     */
    fun mouseMove(x: Int, y: Int, event: TerminalMouseEvent) {}

}

interface TerminalKeyboard {
    /**
     * 键盘按下事件，这个方法和 [keyReleased] 是匹配的。
     *
     * 如果调用一次本方法，那么必须也要调用一次 [keyReleased] 。否则在 [DataKey.AutoRepeatKeys] 场景下可能不工作。
     */
    fun keyPressed(event: TerminalKeyEvent) {}

    /**
     * Type...
     */
    fun keyTyped(event: TerminalKeyEvent) {}

    /**
     * 键盘松开事件
     */
    fun keyReleased(event: TerminalKeyEvent) {}
}

/**
 * 可选择的 Terminal
 */
interface TerminalSelectable {

    /**
     * 开始选择
     */
    fun beginSelect(position: Position, event: TerminalEvent) {}

    /**
     * 选择
     */
    fun select(position: Position, event: TerminalEvent) {}

    /**
     * 是否正在选择中
     */
    fun isSelecting(): Boolean {
        return false
    }

    /**
     * 结束选择
     */
    fun endSelect(position: Position, event: TerminalEvent) {}


}