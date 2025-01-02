package app.termora

import org.apache.sshd.common.config.keys.KeyUtils
import kotlin.test.Test
import kotlin.test.assertEquals

class KeyUtilsTest {
    @Test
    fun test() {
        assertEquals(KeyUtils.getKeySize(KeyUtils.generateKeyPair("ssh-rsa", 1024).private), 1024)
        assertEquals(KeyUtils.getKeySize(KeyUtils.generateKeyPair("ssh-rsa", 1024).public), 1024)
    }
}