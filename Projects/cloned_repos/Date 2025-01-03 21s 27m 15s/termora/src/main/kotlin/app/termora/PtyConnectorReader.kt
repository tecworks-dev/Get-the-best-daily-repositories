package app.termora

import app.termora.terminal.PtyConnector
import app.termora.terminal.Terminal
import kotlinx.coroutines.delay
import javax.swing.SwingUtilities
import kotlin.time.Duration.Companion.milliseconds

class PtyConnectorReader(
    private val ptyConnector: PtyConnector,
    private val terminal: Terminal,
) {

    suspend fun start() {
        var i: Int
        val buffer = CharArray(1024 * 8)
        while ((ptyConnector.read(buffer).also { i = it }) != -1) {
            if (i == 0) {
                delay(10.milliseconds)
                continue
            }
            val text = String(buffer, 0, i)
            SwingUtilities.invokeLater { terminal.write(text) }
        }
    }

}