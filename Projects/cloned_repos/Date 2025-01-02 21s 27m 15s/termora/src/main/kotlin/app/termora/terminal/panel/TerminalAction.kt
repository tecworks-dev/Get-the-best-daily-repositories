package app.termora.terminal.panel

import java.awt.event.KeyEvent
import javax.swing.KeyStroke

abstract class TerminalAction(private val keyStroke: KeyStroke) : TerminalPredicateAction {

    override fun test(keyStroke: KeyStroke, e: KeyEvent): Boolean {
        return keyStroke == this.keyStroke
    }
}