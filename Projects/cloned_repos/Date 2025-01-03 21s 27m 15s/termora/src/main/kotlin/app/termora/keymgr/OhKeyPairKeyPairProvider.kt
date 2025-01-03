package app.termora.keymgr

import app.termora.AES.decodeBase64
import app.termora.RSA
import org.apache.sshd.common.keyprovider.AbstractResourceKeyPairProvider
import org.apache.sshd.common.session.SessionContext
import org.slf4j.LoggerFactory
import java.security.Key
import java.security.KeyPair
import java.security.PrivateKey
import java.security.PublicKey
import java.util.*
import java.util.concurrent.ConcurrentHashMap

class OhKeyPairKeyPairProvider(private val id: String) : AbstractResourceKeyPairProvider<String>() {
    companion object {
        private val log = LoggerFactory.getLogger(OhKeyPairKeyPairProvider::class.java)
        private val cache = ConcurrentHashMap<String, Key>()
    }


    override fun loadKeys(session: SessionContext?): Iterable<KeyPair> {
        val log = OhKeyPairKeyPairProvider.log
        val ohKeyPair = KeyManager.instance.getOhKeyPair(id)
        if (ohKeyPair == null) {
            if (log.isErrorEnabled) {
                log.error("Oh KeyPair [$id] could not be loaded")
            }
            return emptyList()
        }

        return object : Iterable<KeyPair> {
            override fun iterator(): Iterator<KeyPair> {
                val result = kotlin.runCatching {
                    val publicKey = cache.getOrPut(ohKeyPair.publicKey)
                    { RSA.generatePublic(ohKeyPair.publicKey.decodeBase64()) } as PublicKey
                    val privateKey = cache.getOrPut(ohKeyPair.privateKey)
                    { RSA.generatePrivate(ohKeyPair.privateKey.decodeBase64()) } as PrivateKey
                    return@runCatching KeyPair(publicKey, privateKey)
                }
                if (result.isSuccess) {
                    return listOf(result.getOrThrow()).iterator()
                } else if (log.isErrorEnabled) {
                    log.error("Oh KeyPair [$id] could not be loaded.", result.exceptionOrNull())
                }
                return Collections.emptyIterator()
            }
        }
    }
}