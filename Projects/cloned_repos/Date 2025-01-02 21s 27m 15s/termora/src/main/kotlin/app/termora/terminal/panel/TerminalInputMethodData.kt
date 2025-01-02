package app.termora.terminal.panel

import app.termora.terminal.CharBuffer
import app.termora.terminal.TextStyle

data class TerminalInputMethodData(
     val chars: CharBuffer,
     val offset: Int,
) {
    val isTyping get() = !isNoTyping
    val isNoTyping get() = chars.isEmpty()
    val length get() = chars.size

    companion object {
        val Default = TerminalInputMethodData(CharBuffer(charArrayOf(), TextStyle.Default.underline(true)), 0)
    }
}