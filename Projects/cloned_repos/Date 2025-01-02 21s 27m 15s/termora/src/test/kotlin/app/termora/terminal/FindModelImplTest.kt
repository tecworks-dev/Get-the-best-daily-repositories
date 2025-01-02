package app.termora.terminal

import kotlin.test.Test
import kotlin.test.assertEquals

class FindModelImplTest {
    @Test
    fun test() {
        val terminal = VisualTerminal()
        terminal.write("hello world")
        assertEquals(terminal.getFindModel().find("o").size, 2)
    }


    @Test
    fun testMultiline() {
        val terminal = VisualTerminal()
        terminal.write("hello world hello world hello world hello world hello world hello world world 123456789")
        val kind = terminal.getFindModel().find("123456789").first()
        assertEquals(kind.startPosition, Position(1, 79))
        assertEquals(kind.endPosition, Position(2, 7))
    }

    @Test
    fun testChinese() {
        val terminal = VisualTerminal()
        terminal.write("aaaaa.txt")
        val kind = terminal.getFindModel().find("aa.txt").first()
        assertEquals(kind.startPosition, Position(1, 4))
        assertEquals(kind.endPosition, Position(1, 9))
    }
}