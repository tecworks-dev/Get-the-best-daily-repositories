package app.termora.terminal

open class FindKind(
    val startPosition: Position,
    val endPosition: Position
)

abstract class Substr {
    abstract fun size(): Int
    abstract fun at(offset: Int): Char
    open fun next(line: TerminalLine) {

    }

    open fun isEmpty(): Boolean {
        return size() == 0
    }
}

class CharArraySubstr(private val substr: CharArray) : Substr() {
    override fun size(): Int = substr.size
    override fun at(offset: Int): Char {
        return substr[offset]
    }
}

class SubstrFinder(
    private val iterator: Iterator<TerminalLine>,
    private val substr: Substr,
    private val comparator: (Char, Char, Boolean, line: TerminalLine, offset: Int, count: Int) -> Boolean = { a, b, ignoreCase, _, _, _ ->
        a.equals(
            b,
            ignoreCase
        )
    }
) {

    fun find(ignoreCase: Boolean = false): List<FindKind> {
        if (substr.isEmpty()) return emptyList()

        val kinds = mutableListOf<FindKind>()
        var substrIndex = 0
        var startPosition = Position.unknown
        var y = 0
        while (iterator.hasNext()) {
            y++
            val line = iterator.next()

            substr.next(line)

            val chars = line.chars()
            var x = 0
            while (++x <= chars.size) {
                val c = chars[x - 1]
                if (c.first.isNull) break
                if (c.first.isSoftHyphen) continue
                if (!comparator.invoke(substr.at(substrIndex), c.first, ignoreCase, line, x, chars.size)) {
                    // 如果当前字符和要查找的第一个字符匹配，那么回滚重新匹配
                    if (comparator.invoke(substr.at(0), c.first, ignoreCase, line, x, chars.size)) {
                        x -= substrIndex
                    }
                    substrIndex = 0
                    startPosition = Position.unknown
                    continue
                }

                if (startPosition == Position.unknown) {
                    startPosition = Position(y, x)
                }
                ++substrIndex

                // 如果匹配完毕那么表示找到了一个
                if (substrIndex >= substr.size()) {
                    kinds.add(FindKind(startPosition, Position(y, x)))
                    substrIndex = 0
                    startPosition = Position.unknown
                    continue
                }
            }

            // 如果这一行没有换行，那么从新计算
            if (!line.wrapped) {
                substrIndex = 0
                startPosition = Position.unknown
            }
        }

        return kinds
    }

}