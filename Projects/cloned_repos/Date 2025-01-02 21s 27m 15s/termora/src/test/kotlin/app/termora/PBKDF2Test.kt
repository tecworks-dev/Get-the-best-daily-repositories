package app.termora

import org.apache.commons.codec.binary.Hex
import kotlin.test.Test
import kotlin.test.assertEquals

class PBKDF2Test {
    @Test
    fun test() {
        val password = "password"
        assertEquals(
            "72629a41b076e588fba8c71ca37fadc9acdc8e7321b9cb4ea55fd0bf9fe8ed72", Hex.encodeHexString(
                PBKDF2.generateSecret(password.toCharArray(), "salt".toByteArray(), 10000, 256)
            )
        )
    }
}