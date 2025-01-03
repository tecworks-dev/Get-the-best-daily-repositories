package app.termora.terminal


data class HighlighterRange(val start: Position, val end: Position) {
    fun has(pos: Position, cols: Int): Boolean {
        return SelectionModelImpl.isPointInsideArea(start, pos, pos.x, pos.y, cols)
    }
}

interface Highlighter {

    companion object {
        const val HYPERLINK = 1
        const val FIND = 2

    }

    /**
     * 获取样式
     * @param position 字符坐标
     * @param textStyle 原样式
     */
    fun getTextStyle(position: Position, textStyle: TextStyle): TextStyle

    /**
     * 获取标记范围
     */
    fun getHighlighterRange(): HighlighterRange

    /**
     * 终端
     */
    fun getTerminal(): Terminal

    /**
     * 标记
     */
    fun getTag(): Int
}

abstract class TagHighlighter(
    private val range: HighlighterRange,
    private val terminal: Terminal,
    private val tag: Int
) : Highlighter {

    override fun getTextStyle(position: Position, textStyle: TextStyle): TextStyle {
        return textStyle
    }

    override fun getHighlighterRange(): HighlighterRange {
        return range
    }


    override fun getTerminal(): Terminal {
        return terminal
    }

    override fun getTag(): Int {
        return tag
    }

}


abstract class ClickableHighlighter(
    range: HighlighterRange,
    terminal: Terminal,
    tag: Int
) : TagHighlighter(range, terminal, tag) {
    open fun onClicked(position: Position) {}
}

class FindHighlighter(
    range: HighlighterRange,
    terminal: Terminal,
) : TagHighlighter(range, terminal, Highlighter.FIND) {
    private val colorPalette get() = getTerminal().getTerminalModel().getColorPalette()
    override fun getTextStyle(position: Position, textStyle: TextStyle): TextStyle {
        return textStyle.background(colorPalette.getColor(TerminalColor.Find.BACKGROUND))
            .foreground(colorPalette.getColor(TerminalColor.Find.FOREGROUND))
    }
}

class HyperlinkHighlighter(
    range: HighlighterRange,
    terminal: Terminal,
    private val url: String,
    private val onClicked: (position: Position, url: String) -> Unit = { _, _ -> }
) : ClickableHighlighter(range, terminal, Highlighter.HYPERLINK) {
    override fun getTextStyle(position: Position, textStyle: TextStyle): TextStyle {
        return textStyle.copy(
            foreground = getTerminal().getTerminalModel().getColorPalette()
                .getColor(TerminalColor.Basic.HYPERLINK),
            underline = true
        )
    }

    override fun onClicked(position: Position) {
        onClicked.invoke(position, url)
    }
}

/**
 * 标记
 */
interface MarkupModel {


    /**
     * 添加荧光笔
     */
    fun addHighlighter(highlighter: Highlighter)

    /**
     * 移除
     */
    fun removeHighlighter(highlighter: Highlighter)

    /**
     * 移除所有
     * @param tag 为 0 移除所有，不为 0 移除 tag 的
     */
    fun removeAllHighlighters(tag: Int = 0)

    /**
     * 删除某一行的所有荧光笔
     */
    fun removeAllHighlightersInLine(row: Int)

    /**
     * 获取符合的
     */
    fun getHighlighters(position: Position): List<Highlighter>

    /**
     * 获取终端
     */
    fun getTerminal(): Terminal
}