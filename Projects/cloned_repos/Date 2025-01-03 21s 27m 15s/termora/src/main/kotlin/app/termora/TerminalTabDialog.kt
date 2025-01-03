package app.termora

import java.awt.BorderLayout
import java.awt.Dimension
import java.awt.Window
import javax.swing.BorderFactory
import javax.swing.JComponent
import javax.swing.JPanel

class TerminalTabDialog(
    owner: Window,
    size: Dimension,
    private val terminalTab: TerminalTab
) : DialogWrapper(null), Disposable {

    init {
        title = terminalTab.getTitle()
        isModal = false
        isAlwaysOnTop = false
        iconImages = owner.iconImages
        escapeDispose = false
        
        super.setSize(size)

        init()
        setLocationRelativeTo(null)
    }

    override fun createSouthPanel(): JComponent? {
        return null
    }

    override fun createCenterPanel(): JComponent {
        val panel = JPanel(BorderLayout())
        panel.add(terminalTab.getJComponent(), BorderLayout.CENTER)
        panel.border = BorderFactory.createMatteBorder(1, 0, 0, 0, DynamicColor.BorderColor)
        return panel
    }

    override fun dispose() {
        Disposer.dispose(this)
        super<DialogWrapper>.dispose()
    }

}