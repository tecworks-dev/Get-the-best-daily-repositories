package app.termora.terminal

import kotlin.math.max
import kotlin.math.min

open class TabulatorImpl(private val terminal: Terminal) : Tabulator {
    internal val tabStops = sortedSetOf<Int>()

    private val cols get() = terminal.getTerminalModel().getCols()


    init {
        for (i in 8 until cols step 8) {
            tabStops.add(i)
        }

        terminal.getTerminalModel().addDataListener(object : DataListener {
            override fun onChanged(key: DataKey<*>, data: Any) {
                if (key == TerminalModel.Resize) {
                    clearAllTabStops()
                    for (i in 8 until cols step 8) {
                        tabStops.add(i)
                    }
                }
            }
        })
    }


    override fun clearTabStop(x: Int) {
        tabStops.remove(x)
    }

    override fun clearAllTabStops() {
        tabStops.clear()
    }

    override fun nextTab(x: Int): Int {
        var tabStop = Int.MAX_VALUE


        // Search for the first tab stop after the given position...
        val tailSet = tabStops.tailSet(x + 1)
        if (tailSet.isNotEmpty()) {
            tabStop = tailSet.first()
        }


        // Don't go beyond the end of the line...
        return min(tabStop, (cols))
    }

    override fun previousTab(x: Int): Int {
        val headSet = tabStops.headSet(x)
        var tabStop = 1

        if (!headSet.isEmpty()) {
            tabStop = headSet.last()
        }

        // Don't go beyond the start of the line...
        return max(1, tabStop)
    }

    override fun setTabStop(x: Int) {
        tabStops.add(x)
    }

    override fun getTerminal(): Terminal {
        return terminal
    }
}