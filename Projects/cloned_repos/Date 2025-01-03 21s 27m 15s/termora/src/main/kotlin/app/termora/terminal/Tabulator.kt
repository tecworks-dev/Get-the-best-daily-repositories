// https://github.com/JetBrains/jediterm/blob/c1cb74df9df3b529b5715810eaa4a61a21570a25/LICENSE-APACHE-2.0.txt
// https://github.com/JetBrains/jediterm/blob/c1cb74df9df3b529b5715810eaa4a61a21570a25/core/src/com/jediterm/terminal/model/Tabulator.java

package app.termora.terminal

/**
 *
 * Provides a tabulator that keeps track of the tab stops of a terminal.
 */
interface Tabulator {
    /**
     * Clears the tab stop at the given position.
     *
     * @param x
     * the column position used to determine the next tab stop, > 0.
     */
    fun clearTabStop(x: Int)

    /**
     * Clears all tab stops.
     */
    fun clearAllTabStops()

    /**
     * Returns the next tab stop that is at or after the given position.
     *
     * @param x
     * the column position used to determine the next tab stop, >= 0.
     * @return the next tab stop, >= 0.
     */
    fun nextTab(x: Int): Int

    /**
     * Returns the previous tab stop that is before the given position.
     *
     * @param x
     * the column position used to determine the previous tab stop, >= 0.
     * @return the previous tab stop, >= 0.
     */
    fun previousTab(x: Int): Int

    /**
     * Sets the tab stop to the given position.
     *
     * @param x
     * the position of the (new) tab stop, > 0.
     */
    fun setTabStop(x: Int)

    /**
     * 获取终端
     */
    fun getTerminal(): Terminal
}