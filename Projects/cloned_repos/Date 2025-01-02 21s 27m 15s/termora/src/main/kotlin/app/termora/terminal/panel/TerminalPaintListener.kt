package app.termora.terminal.panel

import app.termora.terminal.Terminal
import java.awt.Graphics
import java.util.*


/**
 * 渲染事件
 */
interface TerminalPaintListener : EventListener {
    fun before(
        offset: Int,
        count: Int,
        g: Graphics,
        terminalPanel: TerminalPanel,
        terminalDisplay: TerminalDisplay,
        terminal: Terminal
    ) {
    }

    fun after(
        offset: Int,
        count: Int,
        g: Graphics,
        terminalPanel: TerminalPanel,
        terminalDisplay: TerminalDisplay,
        terminal: Terminal
    ) {
    }
}