package app.termora.terminal

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TextStyleTest {
    @Test
    fun test() {
        var textStyle = TextStyle()
        assertFalse(textStyle.bold)
        assertTrue(textStyle.bold(true).bold)


        textStyle = textStyle.foreground(255 * 255 * 2)
        textStyle = textStyle.background(255 * 2)

        assertEquals(textStyle.foreground, 255 * 255 * 2)
        assertEquals(textStyle.background, 255 * 2)
        assertFalse(textStyle.italic)

        textStyle = textStyle.italic(true)
        assertTrue(textStyle.italic)

    }
}