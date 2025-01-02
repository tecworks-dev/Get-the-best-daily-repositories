package app.termora.terminal

data class PositionRange(val start: Position, val end: Position)

data class Position(
    /**
     * Y row
     */
    val y: Int,
    /**
     * X col
     */
    val x: Int,
) {

    companion object {
        val unknown: Position get() = Position(y = -1, x = -1)
    }

    fun isValid(): Boolean {
        return this != unknown
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Position) return true
        if (x != other.x) return false
        if (y != other.y) return false

        return true
    }

    override fun hashCode(): Int {
        var result = x.hashCode()
        result = 31 * result + y.hashCode()
        return result
    }


}

