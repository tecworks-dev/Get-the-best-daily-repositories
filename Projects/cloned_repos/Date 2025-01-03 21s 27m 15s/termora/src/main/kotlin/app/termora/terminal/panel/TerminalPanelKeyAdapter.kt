package app.termora.terminal.panel

import app.termora.terminal.PtyConnector
import app.termora.terminal.Terminal
import java.awt.event.KeyAdapter
import java.awt.event.KeyEvent
import javax.swing.KeyStroke

class TerminalPanelKeyAdapter(
    private val terminalPanel: TerminalPanel,
    private val terminal: Terminal,
    private val ptyConnector: PtyConnector
) :
    KeyAdapter() {

    override fun keyTyped(e: KeyEvent) {
        if (Character.isISOControl(e.keyChar)) {
            return
        }

        terminal.getSelectionModel().clearSelection()
        ptyConnector.write("${e.keyChar}")
        terminal.getScrollingModel().scrollTo(Int.MAX_VALUE)

    }

    override fun keyPressed(e: KeyEvent) {
        if (e.isConsumed) return

        // remove all toast
        if (e.keyCode == KeyEvent.VK_ESCAPE) {
            terminalPanel.hideToast()
        }

        val keyStroke = KeyStroke.getKeyStrokeForEvent(e)
        for (action in terminalPanel.getTerminalActions()) {
            if (action.test(keyStroke, e)) {
                action.actionPerformed(e)
                return
            }
        }

        val encode = terminal.getKeyEncoder().encode(AWTTerminalKeyEvent(e))
        if (encode.isNotEmpty()) {
            ptyConnector.write(encode)
        }

        if (Character.isISOControl(e.keyChar)) {
            terminal.getSelectionModel().clearSelection()
            // 如果不为空表示已经发送过了，所以这里为空的时候再发送
            if (encode.isEmpty()) {
                ptyConnector.write("${e.keyChar}")
            }
            terminal.getScrollingModel().scrollTo(Int.MAX_VALUE)
        }

    }
}