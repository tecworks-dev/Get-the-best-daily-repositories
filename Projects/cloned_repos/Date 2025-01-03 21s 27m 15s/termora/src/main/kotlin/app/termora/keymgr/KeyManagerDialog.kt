package app.termora.keymgr

import app.termora.DialogWrapper
import app.termora.I18n
import app.termora.OptionPane
import java.awt.Dimension
import java.awt.Window
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import javax.swing.JComponent
import javax.swing.SwingUtilities
import javax.swing.UIManager

class KeyManagerDialog(
    owner: Window,
    private val selectMode: Boolean = false,
    size: Dimension = Dimension(UIManager.getInt("Dialog.width"), UIManager.getInt("Dialog.height")),
) : DialogWrapper(owner) {
    private val keyManagerPanel by lazy { KeyManagerPanel() }
    var ok: Boolean = false

    init {
        super.setSize(size.width, size.height)
        isModal = true
        title = I18n.getString("termora.keymgr.title")
        setLocationRelativeTo(null)

        init()

        if (selectMode) {
            keyManagerPanel.keyPairTable.addMouseListener(object : MouseAdapter() {
                override fun mouseClicked(e: MouseEvent) {
                    if (e.clickCount % 2 == 0 && SwingUtilities.isLeftMouseButton(e)) {
                        if (keyManagerPanel.keyPairTable.selectedRowCount > 0) {
                            SwingUtilities.invokeLater { doOKAction() }
                        }
                    }
                }
            })
        }
    }

    override fun createCenterPanel(): JComponent {
        return keyManagerPanel
    }

    override fun createSouthPanel(): JComponent? {
        return if (selectMode) super.createSouthPanel() else null
    }

    override fun doOKAction() {
        if (selectMode) {
            if (keyManagerPanel.keyPairTable.selectedRowCount < 1) {
                OptionPane.showMessageDialog(this, "Please select a Key")
                return
            }
        }
        ok = true
        super.doOKAction()
    }

    fun getLasOhKeyPair(): OhKeyPair? {
        if (keyManagerPanel.keyPairTable.selectedRowCount > 0) {
            val row = keyManagerPanel.keyPairTable.selectedRows.last()
            return keyManagerPanel.keyPairTableModel.getOhKeyPair(row)
        }
        return null
    }

}