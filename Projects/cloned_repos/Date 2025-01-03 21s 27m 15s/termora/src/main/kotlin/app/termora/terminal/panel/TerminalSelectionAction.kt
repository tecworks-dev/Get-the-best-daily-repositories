package app.termora.terminal.panel

import app.termora.terminal.Terminal
import java.awt.event.KeyEvent
import javax.swing.KeyStroke

/**
 * https://learn.microsoft.com/zh-cn/windows/terminal/selection
 */
class TerminalSelectionAction(private val terminal: Terminal) : TerminalPredicateAction {

    override fun actionPerformed(e: KeyEvent) {
    }

    override fun test(keyStroke: KeyStroke, e: KeyEvent): Boolean {
        return false
    }


}