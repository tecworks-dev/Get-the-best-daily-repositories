package app.termora.terminal.panel

import app.termora.I18n
import app.termora.terminal.*
import kotlin.reflect.cast

class TerminalPanelRepaintListener(
    private val terminalPanel: TerminalPanel,
) : DataListener {
    companion object {
        private val keys = setOf(
            TerminalPanel.Debug, SelectionModel.Selection,
            VisualTerminal.Written, DataKey.CursorStyle,
            ScrollingModel.Scroll, TerminalModel.Resize,
        )
    }

    override fun onChanged(key: DataKey<*>, data: Any) {
        if (keys.contains(key)) {
            terminalPanel.repaintImmediate()
        }

        if (key == TerminalModel.Resize) {
            if (terminalPanel.resizeToast) {
                val size = TerminalModel.Resize.clazz.cast(data).newSize
                terminalPanel.toast(I18n.getString("termora.terminal.size", size.cols, size.rows))
            }
        }
    }
}