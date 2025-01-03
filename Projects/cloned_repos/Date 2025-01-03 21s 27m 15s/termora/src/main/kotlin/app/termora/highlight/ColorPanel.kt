package app.termora.highlight

import java.awt.Color
import javax.swing.JPanel

class ColorPanel : JPanel {
    var color: Color = Color.WHITE
        set(value) {
            background = value
            val old = field
            field = value
            firePropertyChange("color", old, value)
        }
    var colorIndex = -1

    constructor(color: Color) : super() {
        this.color = color
    }
}