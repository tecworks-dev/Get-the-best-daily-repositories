package app.termora.terminal

open class TerminalEvent(val modifiers: Int) {

    companion object {
        const val SHIFT_MASK = 4
        const val META_MASK = 8
        const val CTRL_MASK = 16
        const val ALT_MASK = 32
        const val ALT_GRAPH_MASK = 64
    }

    private var consumed = false
    val isConsumed get() = consumed

    val isCtrlDown get() = (modifiers and CTRL_MASK) == CTRL_MASK
    val isShiftDown get() = (modifiers and SHIFT_MASK) == SHIFT_MASK
    val isMetaDown get() = (modifiers and META_MASK) == META_MASK
    val isAltDown get() = (modifiers and ALT_MASK) == ALT_MASK
    val isAltGraphDown get() = (modifiers and ALT_GRAPH_MASK) == ALT_GRAPH_MASK

    fun consume() {
        consumed = true
    }


}

class HyperlinkEvent(val url: String) : TerminalEvent(0)