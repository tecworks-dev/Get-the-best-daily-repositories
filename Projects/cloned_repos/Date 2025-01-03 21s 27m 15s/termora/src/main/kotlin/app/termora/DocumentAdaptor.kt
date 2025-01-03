package app.termora

import javax.swing.event.DocumentEvent
import javax.swing.event.DocumentListener

abstract class DocumentAdaptor : DocumentListener {
    override fun insertUpdate(e: DocumentEvent) {
        changedUpdate(e)
    }

    override fun removeUpdate(e: DocumentEvent) {
        changedUpdate(e)
    }

    override fun changedUpdate(e: DocumentEvent) {

    }
}