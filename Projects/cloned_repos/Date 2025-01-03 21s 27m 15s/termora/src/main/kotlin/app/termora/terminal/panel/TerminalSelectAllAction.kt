package app.termora.terminal.panel

import app.termora.terminal.Position
import app.termora.terminal.Terminal
import java.awt.Toolkit
import java.awt.event.KeyEvent
import javax.swing.KeyStroke

class TerminalSelectAllAction(private val terminal: Terminal) : TerminalAction(
    KeyStroke.getKeyStroke(
        KeyEvent.VK_A,
        Toolkit.getDefaultToolkit().menuShortcutKeyMaskEx
    )
) {
    override fun actionPerformed(e: KeyEvent) {
        terminal.getSelectionModel().setSelection(
            Position(y = 1, x = 1),
            Position(y = terminal.getDocument().getLineCount(), x = terminal.getTerminalModel().getCols())
        )
    }
}