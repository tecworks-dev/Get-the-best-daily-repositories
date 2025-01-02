package app.termora

import java.text.BreakIterator
import kotlin.test.Test

class BreakIteratorTest {
    @Test
    fun test() {
        val text = "Hello World"
        val breakIterator = BreakIterator.getCharacterInstance()
        breakIterator.setText(text)
        var start = breakIterator.first()
        var end = breakIterator.next()
        while (end != BreakIterator.DONE) {
            println(text.substring(start, end))
            start = end
            end = breakIterator.next()
        }

    }


    @Test
    fun testChinese() {
        val text = "你好中国 hello 123 @！。"
        val breakIterator = BreakIterator.getCharacterInstance()
        breakIterator.setText(text)
        var start = breakIterator.first()
        var end = breakIterator.next()
        while (end != BreakIterator.DONE) {
            println(text.substring(start, end))
            start = end
            end = breakIterator.next()
        }
    }

    @Test
    fun testEmoji() {
        val text = "⌚️哈哈😂123🀄️"
        val breakIterator = BreakIterator.getCharacterInstance()
        breakIterator.setText(text)
        var start = breakIterator.first()
        var end = breakIterator.next()
        while (end != BreakIterator.DONE) {
            println(text.substring(start, end))
            start = end
            end = breakIterator.next()
        }
    }
}