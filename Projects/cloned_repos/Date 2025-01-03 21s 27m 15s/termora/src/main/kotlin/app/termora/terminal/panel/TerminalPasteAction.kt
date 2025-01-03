package app.termora.terminal.panel

import com.formdev.flatlaf.util.SystemInfo
import org.slf4j.LoggerFactory
import java.awt.datatransfer.DataFlavor
import java.awt.event.KeyEvent
import javax.swing.KeyStroke

class TerminalPasteAction(private val terminalPanel: TerminalPanel) : TerminalAction(
    KeyStroke.getKeyStroke(KeyEvent.VK_V, terminalPanel.toolkit.menuShortcutKeyMaskEx)
) {
    companion object {
        private val log = LoggerFactory.getLogger(TerminalPasteAction::class.java)
    }

    private val systemClipboard get() = terminalPanel.toolkit.systemClipboard

    override fun actionPerformed(e: KeyEvent) {
        if (systemClipboard.isDataFlavorAvailable(DataFlavor.stringFlavor)) {
            val text = systemClipboard.getData(DataFlavor.stringFlavor)
            if (text is String) {
                terminalPanel.paste(text)
                if (log.isTraceEnabled) {
                    log.info("Paste {}", text)
                }
            }
        }
    }

    override fun test(keyStroke: KeyStroke, e: KeyEvent): Boolean {
        if (!SystemInfo.isMacOS) {
            return false
        }
        return super.test(keyStroke, e)
    }

}