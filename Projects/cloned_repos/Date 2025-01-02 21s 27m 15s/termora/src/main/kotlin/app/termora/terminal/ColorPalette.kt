package app.termora.terminal


interface TerminalColor {

    companion object {
        private fun generator(): TerminalColor {
            return object : TerminalColor {}
        }
    }

    object Find {
        val BACKGROUND = generator()
        val FOREGROUND = generator()
    }

    object Basic {
        val BACKGROUND = generator()
        val FOREGROUND = generator()

        val SELECTION_FOREGROUND = generator()
        val SELECTION_BACKGROUND = generator()

        val HYPERLINK = generator()

    }

    object Cursor {
        val BACKGROUND = generator()
    }

    object Normal {
        val BLACK = generator()
        val RED = generator()
        val GREEN = generator()
        val YELLOW = generator()
        val BLUE = generator()
        val MAGENTA = generator()
        val CYAN = generator()
        val WHITE = generator()
    }

    object Bright {
        val BLACK = generator()
        val RED = generator()
        val GREEN = generator()
        val YELLOW = generator()
        val BLUE = generator()
        val MAGENTA = generator()
        val CYAN = generator()
        val WHITE = generator()
    }

}

interface ColorPalette {
    fun getTerminal(): Terminal

    /**
     * @return rgb
     */
    fun getColor(color: TerminalColor): Int

    /**
     * 根据索引获取颜色
     * @return rgb
     */
    fun getXTerm256Color(index: Int): Int

}