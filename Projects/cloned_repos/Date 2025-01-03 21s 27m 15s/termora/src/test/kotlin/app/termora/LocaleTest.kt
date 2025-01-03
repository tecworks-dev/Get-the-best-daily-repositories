package app.termora

import org.apache.commons.lang3.LocaleUtils
import java.util.*
import kotlin.test.Test
import kotlin.test.assertEquals

class LocaleTest {
    @Test
    fun test() {
        assertEquals(Locale.of("zh_CN").toString(), "zh_cn")
        assertEquals(Locale.of("zh", "CN").toString(), "zh_CN")
        assertEquals(LocaleUtils.toLocale("zh_CN").toString(), "zh_CN")
    }
}