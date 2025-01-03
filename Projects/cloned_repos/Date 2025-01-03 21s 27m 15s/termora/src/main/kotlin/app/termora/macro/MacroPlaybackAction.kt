package app.termora.macro

import app.termora.Actions
import app.termora.AnAction
import app.termora.I18n
import org.jdesktop.swingx.action.ActionManager
import java.awt.event.ActionEvent

class MacroPlaybackAction : AnAction(
    I18n.getString("termora.macro.playback"),
) {
    private val macroAction get() = ActionManager.getInstance().getAction(Actions.MACRO) as MacroAction?
    private val macroManager get() = MacroManager.instance
    override fun actionPerformed(evt: ActionEvent) {
        val macros = macroManager.getMacros().sortedByDescending { it.sort }
        if (macros.isEmpty() || macroAction == null) {
            return
        }
        macroAction?.runMacro(macros.first())
    }

    override fun isEnabled(): Boolean {
        if (macroAction == null) {
            return false
        }
        return macroManager.getMacros().isNotEmpty()
    }
}