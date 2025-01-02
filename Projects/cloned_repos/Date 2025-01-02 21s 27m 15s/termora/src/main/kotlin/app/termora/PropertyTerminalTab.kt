package app.termora

import java.beans.PropertyChangeEvent
import java.beans.PropertyChangeListener

abstract class PropertyTerminalTab : TerminalTab {
    protected val listeners = mutableListOf<PropertyChangeListener>()
    var hasFocus = false
        protected set

    override fun addPropertyChangeListener(listener: PropertyChangeListener) {
        listeners.add(listener)
    }

    override fun removePropertyChangeListener(listener: PropertyChangeListener) {
        listeners.remove(listener)
    }

    protected fun firePropertyChange(event: PropertyChangeEvent) {
        listeners.forEach { l -> l.propertyChange(event) }
    }

    override fun onGrabFocus() {
        hasFocus = true
    }

    override fun onLostFocus() {
        hasFocus = false
    }


}