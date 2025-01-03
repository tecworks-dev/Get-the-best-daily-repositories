package app.termora.findeverywhere

import app.termora.I18n

class SettingsFindEverywhereProvider : FindEverywhereProvider {


    override fun find(pattern: String): List<FindEverywhereResult> {
        return emptyList()
    }


    override fun group(): String {
        return I18n.getString("termora.find-everywhere.groups.settings")
    }

}