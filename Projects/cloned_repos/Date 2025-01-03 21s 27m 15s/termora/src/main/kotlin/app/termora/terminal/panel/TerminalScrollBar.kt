package app.termora.terminal.panel

import app.termora.terminal.Terminal
import app.termora.terminal.TerminalColor
import com.formdev.flatlaf.ui.FlatScrollBarUI
import java.awt.Color
import java.awt.Graphics
import java.awt.Rectangle
import javax.swing.JComponent
import javax.swing.JScrollBar
import javax.swing.plaf.ScrollBarUI
import kotlin.math.ceil
import kotlin.math.max

class TerminalScrollBar(
    private val terminalPanel: TerminalPanel,
    private val terminalFindPanel: TerminalFindPanel,
    private val terminal: Terminal
) : JScrollBar() {
    private val colorPalette get() = terminal.getTerminalModel().getColorPalette()
    private val myUI = MyScrollBarUI()

    init {
        setUI(myUI)
    }

    override fun setUI(ui: ScrollBarUI) {
        super.setUI(myUI)
    }

    private fun drawFindMap(g: Graphics, trackBounds: Rectangle) {
        if (!terminalPanel.findMap) return
        val kinds = terminalFindPanel.kinds
        if (kinds.isEmpty()) return

        val averageCharWidth = terminalPanel.getAverageCharWidth() * 2
        val count = max(terminal.getDocument().getLineCount(), terminal.getTerminalModel().getRows())
        // 高度/总行数 就可以计算出标记的平均高度
        val lineHeight = max(ceil(1.0 * trackBounds.height / count).toInt(), 1)

        val rows = linkedSetOf<Int>()
        for (kind in kinds) {
            rows.add(kind.startPosition.y)
            rows.add(kind.endPosition.y)
        }

        g.color = Color(colorPalette.getColor(TerminalColor.Find.BACKGROUND))

        for (row in rows) {
            // 计算行应该在总行的哪个位置
            val n = row * 1.0 / count
            // 根据比例计算出标记的位置
            val y = max(ceil(trackBounds.height * n).toInt() - lineHeight, 0)
            g.fillRect(trackBounds.width - averageCharWidth, y, averageCharWidth, lineHeight)
        }

    }

    private inner class MyScrollBarUI : FlatScrollBarUI() {
        override fun paintTrack(g: Graphics, c: JComponent, trackBounds: Rectangle) {
            super.paintTrack(g, c, trackBounds)
            drawFindMap(g, trackBounds)
        }
    }
}