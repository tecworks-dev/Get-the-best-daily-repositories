package app.termora.terminal.panel

import app.termora.terminal.ClickableHighlighter
import app.termora.terminal.Terminal
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import javax.swing.SwingUtilities

/**
 * 超链接点击时
 */
class TerminalPanelMouseHyperlinkAdapter(
    private val terminalPanel: TerminalPanel,
    private val terminal: Terminal,
) : MouseAdapter() {

    override fun mouseClicked(e: MouseEvent) {
        if (SwingUtilities.isLeftMouseButton(e)) {
            val position = terminalPanel.pointToPosition(e.point)
            for (highlighter in terminal.getMarkupModel().getHighlighters(position)) {
                if (highlighter is ClickableHighlighter) {
                    highlighter.onClicked(position)
                }
            }
        }
    }


}