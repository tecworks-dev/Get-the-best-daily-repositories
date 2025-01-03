package app.termora.terminal

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min


open class ScrollingModelImpl(private val terminal: Terminal) : ScrollingModel {

    protected class ScrollModel(private val terminal: Terminal) {
        var offset = 0
            set(value) {
                field = max(value, 0)
            }

        var stick = true

        /**
         * 允许的最大偏移量，也就是总行数
         */
        val maxVerticalScrollOffset
            get() = terminal.getDocument().getCurrentTerminalLineBuffer()
                .getBufferCount()


    }

    /**
     * [DataKey.AlternateScreenBuffer]
     */
    private val screenModel = ScrollModel(terminal)
    private val scrollModel = ScrollModel(terminal)


    override fun getTerminal(): Terminal {
        return terminal
    }

    override fun getVerticalScrollOffset(): Int {
        return min(getScrollModel().offset, getMaxVerticalScrollOffset())
    }

    override fun getMaxVerticalScrollOffset(): Int {
        return getScrollModel().maxVerticalScrollOffset
    }

    override fun getAlternateScreenBufferScrollingModel(): ScrollingModel {
        return object : ScrollingModelImpl(terminal) {
            override fun getScrollModel(): ScrollModel {
                return screenModel
            }

            override fun getAlternateScreenBufferScrollingModel(): ScrollingModel {
                return this
            }

            override fun getNonAlternateScreenBufferScrollingModel(): ScrollingModel {
                return this@ScrollingModelImpl.getNonAlternateScreenBufferScrollingModel()
            }
        }
    }

    override fun getNonAlternateScreenBufferScrollingModel(): ScrollingModel {
        return object : ScrollingModelImpl(terminal) {
            override fun getScrollModel(): ScrollModel {
                return scrollModel
            }

            override fun getAlternateScreenBufferScrollingModel(): ScrollingModel {
                return this@ScrollingModelImpl.getAlternateScreenBufferScrollingModel()
            }

            override fun getNonAlternateScreenBufferScrollingModel(): ScrollingModel {
                return this
            }
        }
    }

    override fun isStick(): Boolean {
        return getScrollModel().stick
    }

    override fun canVerticalScroll(): Boolean {
        return getMaxVerticalScrollOffset() > 0
    }

    protected open fun getScrollModel(): ScrollModel {
        return if (terminal.getTerminalModel().isAlternateScreenBuffer()) screenModel else scrollModel
    }

    override fun scrollTo(offset: Int) {
        val model = getScrollModel()
        val maxVerticalScrollOffset = getMaxVerticalScrollOffset()

        if (offset > maxVerticalScrollOffset) {
            model.offset = maxVerticalScrollOffset
        } else {
            model.offset = offset
        }

        // stick
        model.stick = model.offset == maxVerticalScrollOffset

        // 发布滚动事件
        terminal.getTerminalModel().setData(ScrollingModel.Scroll, model.offset)

    }

    override fun scrollToRow(row: Int) {
        if (isInVisibleArea(row)) {
            return
        }

        val start = getVerticalScrollOffset()
        val end = start + terminal.getTerminalModel().getRows()

        // 向上
        if (row - 1 < start) {
            scrollTo(start - abs(row - start - 1))
        } else {
            scrollTo(start + abs(row - end))
        }

    }

    override fun isInVisibleArea(row: Int): Boolean {
        val rows = terminal.getTerminalModel().getRows()
        val lineCount = max(terminal.getDocument().getLineCount(), rows)
        val offset = max(getVerticalScrollOffset(), 0)
        val end = min(lineCount, offset + rows)
        return row in (offset + 1)..end
    }


}