package app.termora.findeverywhere

import java.awt.event.ActionEvent

class GroupFindEverywhereResult(private val groupName: String) : FindEverywhereResult {
    override fun actionPerformed(e: ActionEvent) {
        throw UnsupportedOperationException()
    }

    override fun toString(): String {
        return groupName
    }
}