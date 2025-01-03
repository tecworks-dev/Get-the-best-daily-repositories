package app.termora

import com.formdev.flatlaf.extras.components.FlatTextField
import org.apache.commons.lang3.StringUtils
import java.awt.Window
import java.awt.event.KeyAdapter
import java.awt.event.KeyEvent
import javax.swing.BorderFactory
import javax.swing.JComponent
import javax.swing.UIManager

class InputDialog(
    owner: Window,
    title: String,
    text: String = StringUtils.EMPTY,
    placeholderText: String = StringUtils.EMPTY
) : DialogWrapper(owner) {
    private val textField = FlatTextField()
    private var text: String? = null

    init {
        setSize(340, 60)
        setLocationRelativeTo(owner)

        super.setTitle(title)

        isResizable = false
        isModal = true
        controlsVisible = false
        titleBarHeight = UIManager.getInt("TabbedPane.tabHeight") * 0.8f


        textField.placeholderText = placeholderText
        textField.text = text
        textField.addKeyListener(object : KeyAdapter() {
            override fun keyPressed(e: KeyEvent) {
                if (e.keyCode == KeyEvent.VK_ENTER) {
                    if (textField.text.isBlank()) {
                        return
                    }
                    doOKAction()
                }
            }
        })

        init()
    }

    override fun createCenterPanel(): JComponent {
        textField.background = UIManager.getColor("window")
        textField.border = BorderFactory.createEmptyBorder(0, 13, 0, 13)

        return textField
    }

    fun getText(): String? {
        isVisible = true
        return text
    }

    override fun doCancelAction() {
        text = null
        super.doCancelAction()
    }

    override fun doOKAction() {
        text = textField.text
        super.doOKAction()
    }

    override fun createSouthPanel(): JComponent? {
        return null
    }
}