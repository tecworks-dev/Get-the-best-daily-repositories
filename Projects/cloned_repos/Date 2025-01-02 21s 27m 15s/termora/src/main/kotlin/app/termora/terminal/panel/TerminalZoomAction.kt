package app.termora.terminal.panel

import app.termora.TerminalPanelFactory
import app.termora.db.Database
import java.awt.event.KeyEvent
import javax.swing.KeyStroke

abstract class TerminalZoomAction(keyStroke: KeyStroke) : TerminalAction(keyStroke) {
    protected val fontSize get() = Database.instance.terminal.fontSize

    override fun actionPerformed(e: KeyEvent) {
        if (!zoom()) return
        TerminalPanelFactory.instance.fireResize()
    }

    abstract fun zoom(): Boolean
}