package app.termora.keymgr

import app.termora.db.Database
import org.slf4j.LoggerFactory

class KeyManager private constructor() {
    companion object {
        private val log = LoggerFactory.getLogger(KeyManager::class.java)
        val instance by lazy { KeyManager() }
    }

    private val keyPairs = mutableSetOf<OhKeyPair>()
    private val database get() = Database.instance

    init {
        keyPairs.addAll(database.getKeyPairs())
    }

    fun addOhKeyPair(keyPair: OhKeyPair) {
        if (keyPair == OhKeyPair.empty) {
            return
        }
        keyPairs.add(keyPair)
        database.addKeyPair(keyPair)
    }

    fun removeOhKeyPair(id: String) {
        keyPairs.removeIf { it.id == id }
        database.removeKeyPair(id)
    }

    fun getOhKeyPairs(): List<OhKeyPair> {
        return keyPairs.sortedBy { it.sort }
    }

    fun getOhKeyPair(id: String): OhKeyPair? {
        return keyPairs.findLast { it.id == id }
    }

    fun removeAll() {
        keyPairs.clear()
        database.removeAllKeyPair()
    }

}