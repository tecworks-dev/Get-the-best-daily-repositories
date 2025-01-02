package app.termora.terminal.panel

import java.awt.event.KeyEvent
import java.util.function.BiPredicate
import javax.swing.KeyStroke

interface TerminalPredicateAction : BiPredicate<KeyStroke, KeyEvent> {
    fun actionPerformed(e: KeyEvent)
}