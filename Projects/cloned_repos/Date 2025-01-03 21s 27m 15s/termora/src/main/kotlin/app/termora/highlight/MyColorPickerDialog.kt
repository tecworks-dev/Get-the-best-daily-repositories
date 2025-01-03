package app.termora.highlight

import app.termora.DialogWrapper
import com.bric.colorpicker.ColorPicker
import java.awt.Color
import java.awt.Window
import javax.swing.JComponent

class MyColorPickerDialog(owner: Window) : DialogWrapper(owner) {
    val colorPicker = ColorPicker()
    var color: Color? = null

    init {
        isModal = true
        title = "Color Picker"
        init()
        pack()
        setLocationRelativeTo(null)
    }

    override fun createCenterPanel(): JComponent {
        return colorPicker
    }

    override fun doOKAction() {
        color = colorPicker.color
        super.doOKAction()
    }

}