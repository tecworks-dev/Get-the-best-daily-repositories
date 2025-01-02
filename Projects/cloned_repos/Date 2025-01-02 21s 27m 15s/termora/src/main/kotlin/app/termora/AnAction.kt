package app.termora

import org.jdesktop.swingx.action.BoundAction
import javax.swing.Icon

abstract class AnAction : BoundAction {

    constructor() : super()
    constructor(icon: Icon) : super() {
        super.putValue(SMALL_ICON, icon)
    }

    constructor(name: String?) : super(name)
    constructor(name: String?, icon: Icon?) : super(name, icon)

}