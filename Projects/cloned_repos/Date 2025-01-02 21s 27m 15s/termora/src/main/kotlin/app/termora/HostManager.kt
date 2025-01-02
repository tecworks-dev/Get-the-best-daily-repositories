package app.termora

import app.termora.db.Database
import java.util.*

interface HostListener : EventListener {
    fun hostAdded(host: Host) {}
    fun hostRemoved(id: String) {}
    fun hostsChanged() {}
}


class HostManager private constructor() {
    companion object {
        val instance by lazy { HostManager() }
    }

    private val database get() = Database.instance
    private val listeners = mutableListOf<HostListener>()

    fun addHost(host: Host, notify: Boolean = true) {
        assertEventDispatchThread()
        database.addHost(host)
        if (notify) listeners.forEach { it.hostAdded(host) }
    }

    fun removeHost(id: String) {
        assertEventDispatchThread()
        database.removeHost(id)
        listeners.forEach { it.hostRemoved(id) }

    }

    fun hosts(): List<Host> {
        return database.getHosts()
            .sortedWith(compareBy<Host> { if (it.protocol == Protocol.Folder) 0 else 1 }.thenBy { it.sort })
    }

    fun removeAll() {
        assertEventDispatchThread()
        database.removeAllHost()
        listeners.forEach { it.hostsChanged() }
    }

    fun addHostListener(listener: HostListener) {
        listeners.add(listener)
    }

    fun removeHostListener(listener: HostListener) {
        listeners.remove(listener)
    }


}