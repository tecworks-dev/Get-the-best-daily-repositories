package app.termora.terminal

import java.io.InputStream
import java.io.OutputStream


abstract class StreamPtyConnector(
    val input: InputStream,
    val output: OutputStream
) : PtyConnector