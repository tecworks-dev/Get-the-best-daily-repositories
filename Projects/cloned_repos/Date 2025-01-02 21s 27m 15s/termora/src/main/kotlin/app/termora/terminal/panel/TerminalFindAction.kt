package app.termora.terminal.panel

import java.awt.Toolkit
import java.awt.event.KeyEvent
import javax.swing.KeyStroke

class TerminalFindAction(private val terminalPanel: TerminalPanel) : TerminalAction(
    KeyStroke.getKeyStroke(
        KeyEvent.VK_F,
Toolkit.getDefaultToolkit().menuShortcutKeyMaskEx    )
) {
    override fun actionPerformed(e: KeyEvent) {
        terminalPanel.showFind()
    }
}