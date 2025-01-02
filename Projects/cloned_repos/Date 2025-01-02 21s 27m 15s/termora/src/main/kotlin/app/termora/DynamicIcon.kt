package app.termora

import com.formdev.flatlaf.extras.FlatSVGIcon

open class DynamicIcon(name: String, private val darkName: String) : FlatSVGIcon(name) {
    constructor(name: String) : this(name, name)

    val dark by lazy { DynamicIcon(darkName, name) }

}
