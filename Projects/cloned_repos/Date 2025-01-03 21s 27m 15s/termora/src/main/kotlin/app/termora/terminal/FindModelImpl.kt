package app.termora.terminal

open class FindModelImpl(private val terminal: Terminal) : FindModel {

    private val document get() = terminal.getDocument()

    override fun getTerminal(): Terminal {
        return terminal
    }

    override fun find(text: String, ignoreCase: Boolean): List<FindKind> {
        if (text.isEmpty()) return emptyList()

        return SubstrFinder(object : Iterator<TerminalLine> {
            private var index = 1
            override fun hasNext(): Boolean {
                return index <= document.getLineCount()
            }

            override fun next(): TerminalLine {
                return document.getLine(index++)
            }

        }, CharArraySubstr(text.toCharArray())).find(ignoreCase)

    }


}