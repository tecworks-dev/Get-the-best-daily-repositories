package app.termora.macro

import app.termora.Actions
import app.termora.AnAction
import app.termora.I18n
import app.termora.Icons
import org.jdesktop.swingx.action.ActionManager
import java.awt.event.ActionEvent
import javax.swing.Icon

class MacroStartRecordingAction(icon: Icon = Icons.rec) : AnAction(
    I18n.getString("termora.macro.start-recording"),
    icon
) {
    private val macroAction get() = ActionManager.getInstance().getAction(Actions.MACRO) as MacroAction?

    override fun actionPerformed(evt: ActionEvent) {
        macroAction?.startRecording()
    }

    override fun isEnabled(): Boolean {
        val action = macroAction ?: return false
        return !action.isRecording
    }
}