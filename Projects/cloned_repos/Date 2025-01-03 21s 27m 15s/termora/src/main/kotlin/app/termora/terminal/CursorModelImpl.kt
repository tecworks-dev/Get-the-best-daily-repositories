package app.termora.terminal

import org.slf4j.LoggerFactory
import kotlin.math.max


open class CursorModelImpl(private val terminal: Terminal) : CursorModel {


    /**
     * [DataKey.AlternateScreenBuffer]
     */
    private var screenPosition = Position(x = 1, y = 1)

    private var position = Position(x = 1, y = 1)

    companion object {
        private val log = LoggerFactory.getLogger(CursorModelImpl::class.java)
    }


    override fun getTerminal(): Terminal {
        return terminal
    }

    override fun getPosition(): Position {
        return if (terminal.getTerminalModel().isAlternateScreenBuffer())
            getAlternateScreenBufferCursorModel().getPosition()
        else
            getNonAlternateScreenBufferCursorModel().getPosition()
    }

    override fun getAlternateScreenBufferCursorModel(): CursorModel {

        return object : CursorModelImpl(terminal) {
            override fun getPosition(): Position {
                return screenPosition
            }

            override fun move(row: Int, col: Int) {
                screenPosition = Position(x = col, y = row)
            }

            override fun getAlternateScreenBufferCursorModel(): CursorModel {
                return this
            }

            override fun getNonAlternateScreenBufferCursorModel(): CursorModel {
                return this@CursorModelImpl.getNonAlternateScreenBufferCursorModel()
            }
        }
    }

    override fun getNonAlternateScreenBufferCursorModel(): CursorModel {
        return object : CursorModelImpl(terminal) {
            override fun getPosition(): Position {
                return position
            }

            override fun move(row: Int, col: Int) {
                position = Position(x = col, y = row)
            }

            override fun getAlternateScreenBufferCursorModel(): CursorModel {
                return this@CursorModelImpl.getAlternateScreenBufferCursorModel()
            }

            override fun getNonAlternateScreenBufferCursorModel(): CursorModel {
                return this
            }
        }
    }

    override fun move(move: CursorMove) {
        move(move, 1)
    }

    override fun move(move: CursorMove, count: Int) {
        if (count <= 0) {
            if (log.isErrorEnabled) {
                log.error("Caret move count $count")
            }
        }

        var position = getPosition()
        position = when (move) {
            CursorMove.RowHome -> {
                position.copy(x = 1)
            }

            CursorMove.RowEnd -> {
                position.copy(x = (getTerminal().getTerminalModel().getCols() - 1))
            }

            CursorMove.Left -> {
                position.copy(x = position.x - count)
            }

            CursorMove.Right -> {
                position.copy(x = position.x + count)
            }

            CursorMove.Down -> {
                position.copy(y = position.y + count)
            }

            CursorMove.Up -> {
                position.copy(y = position.y - count)
            }
        }

        move(row = position.y, col = position.x)
    }

    override fun move(row: Int, col: Int) {
        val newRow = max(row, 1)
        val newCol = max(col, 1)

        if (terminal.getTerminalModel().isAlternateScreenBuffer()) {
            getAlternateScreenBufferCursorModel().move(row = newRow, col = newCol)
            if (log.isTraceEnabled) {
                log.trace(
                    "[Alternate Screen Buffer] Move Caret row:$newRow col:$newCol. Max col: ${
                        terminal.getTerminalModel().getCols()
                    }"
                )
            }

        } else {
            val old = getNonAlternateScreenBufferCursorModel().getPosition()
            getNonAlternateScreenBufferCursorModel().move(row = newRow, col = newCol)
            if (log.isTraceEnabled) {
                log.trace(
                    "Move Caret old row:${old.y} col:${old.x} , row:$newRow col:$newCol. Max col: ${
                        terminal.getTerminalModel().getCols()
                    }"
                )
            }
        }
    }

    override fun getStyle(): CursorStyle {
        return terminal.getTerminalModel().getData(DataKey.CursorStyle)
    }

}