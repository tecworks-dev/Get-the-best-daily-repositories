package app.termora.terminal.panel

import app.termora.db.Database
import java.awt.Toolkit
import java.awt.event.KeyEvent
import javax.swing.KeyStroke

class TerminalZoomResetAction : TerminalZoomAction(
    KeyStroke.getKeyStroke(
        KeyEvent.VK_0,
        Toolkit.getDefaultToolkit().menuShortcutKeyMaskEx
    )
) {

    private val defaultFontSize = 16

    override fun zoom(): Boolean {
        if (fontSize == defaultFontSize) {
            return false
        }
        Database.instance.terminal.fontSize = 16
        return true
    }
}