package app.termora.findeverywhere

import app.termora.Actions
import app.termora.I18n
import app.termora.Icons
import com.formdev.flatlaf.FlatLaf
import org.jdesktop.swingx.action.ActionManager
import javax.swing.Icon

class QuickCommandFindEverywhereProvider : FindEverywhereProvider {


    override fun find(pattern: String): List<FindEverywhereResult> {
        val list = mutableListOf<FindEverywhereResult>()
        ActionManager.getInstance().getAction(Actions.ADD_HOST)?.let {
            list.add(CreateHostFindEverywhereResult())
        }
        return list
    }


    override fun order(): Int {
        return Int.MIN_VALUE
    }

    override fun group(): String {
        return I18n.getString("termora.find-everywhere.groups.quick-actions")
    }

    private class CreateHostFindEverywhereResult : ActionFindEverywhereResult(
        ActionManager.getInstance().getAction(Actions.ADD_HOST)
    ) {
        override fun getIcon(isSelected: Boolean): Icon {
            if (isSelected) {
                if (!FlatLaf.isLafDark()) {
                    return Icons.openNewTab.dark
                }
            }
            return Icons.openNewTab
        }


        override fun toString(): String {
            return I18n.getString("termora.new-host.title")
        }
    }


}