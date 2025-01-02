package app.termora

import kotlin.test.Test
import kotlin.test.assertContentEquals


class AESTest {
    @Test
    fun test() {
        val data = "hello world. 中国 😄".toByteArray()
        val key = AES.randomBytes()
        val iv = AES.randomBytes(16)
        assertContentEquals(AES.CBC.decrypt(key, iv, AES.CBC.encrypt(key, iv, data)), data)
    }


    @Test
    fun testECB() {
        val data = "hello world. 中国 😄".toByteArray()
        val key = AES.randomBytes()
        assertContentEquals(AES.ECB.decrypt(key, AES.ECB.encrypt(key, data)), data)
    }


}