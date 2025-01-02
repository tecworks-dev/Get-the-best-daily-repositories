package app.termora.macro

import app.termora.Actions
import app.termora.AnAction
import app.termora.I18n
import app.termora.Icons
import org.jdesktop.swingx.action.ActionManager
import java.awt.event.ActionEvent
import javax.swing.Icon

class MacroStopRecordingAction(icon: Icon = Icons.stop) : AnAction(
    I18n.getString("termora.macro.stop-recording"),
    icon
) {
    private val macroAction get() = ActionManager.getInstance().getAction(Actions.MACRO) as MacroAction?

    override fun actionPerformed(evt: ActionEvent) {
        macroAction?.stopRecording()
    }

    override fun isEnabled(): Boolean {
        return macroAction?.isRecording ?: false
    }
}