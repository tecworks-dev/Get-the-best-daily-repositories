package app.termora

import javax.swing.JTree
import javax.swing.tree.TreeModel
import javax.swing.tree.TreeNode

object TreeUtils {
    /**
     * 获取子节点
     */
    fun children(
        model: TreeModel,
        parent: Any,
        including: Boolean = true
    ): List<Any> {

        val nodes = mutableListOf<Any>()
        val parents = mutableListOf(parent)

        while (parents.isNotEmpty()) {
            val p = parents.removeFirst()
            for (i in 0 until model.getChildCount(p)) {
                val child = model.getChild(p, i) ?: continue
                nodes.add(child)
                if (including) {
                    parents.add(child)
                }
            }
        }

        return nodes
    }

    fun parents(node: TreeNode): List<Any> {
        val parents = mutableListOf<Any>()
        var p = node.parent
        while (p != null) {
            parents.add(p)
            p = p.parent
        }
        return parents
    }

    fun saveExpansionState(tree: JTree): String {
        val rows = mutableListOf<Int>()
        for (i in 0 until tree.rowCount) {
            if (tree.isExpanded(i)) {
                rows.add(i)
            }
        }
        return rows.joinToString(",")
    }

    fun loadExpansionState(tree: JTree, state: String) {
        if (state.isBlank()) {
            return
        }

        state.split(",")
            .mapNotNull { it.toIntOrNull() }
            .forEach {
                tree.expandRow(it)
            }
    }

    fun expandAll(tree: JTree) {
        var j = tree.rowCount
        var i = 0
        while (i < j) {
            tree.expandRow(i)
            i += 1
            j = tree.rowCount
        }
    }


}