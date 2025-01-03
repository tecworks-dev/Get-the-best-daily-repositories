package app.termora.terminal

@Suppress("MemberVisibilityCanBePrivate")
open class KeyEncoderImpl(private val terminal: Terminal) : KeyEncoder, DataListener {

    private val mapping = mutableMapOf<TerminalKeyEvent, String>()
    private val nothing = String()

    init {

        val terminalModel = terminal.getTerminalModel()
        if (terminalModel.getData(DataKey.ApplicationCursorKeys, false)) {
            arrowKeysApplicationSequences()
        } else {
            arrowKeysAnsiCursorSequences()
        }


        if (terminalModel.getData(DataKey.AlternateKeypad, false)) {
            keypadApplicationSequences()
        } else {
            keypadAnsiSequences()
        }


        putCode(TerminalKeyEvent(keyCode = 8), String(byteArrayOf(127)))

        // Enter
        if (terminalModel.getData(DataKey.AutoNewline, false)) {
            putCode(TerminalKeyEvent(keyCode = 10), encode = "\r\n")
        } else {
            putCode(TerminalKeyEvent(keyCode = 10), encode = "\r")
        }


        // Page Up
        putCode(TerminalKeyEvent(keyCode = 0x21), encode = "${ControlCharacters.ESC}[5~")
        // Page Down
        putCode(TerminalKeyEvent(keyCode = 0x22), encode = "${ControlCharacters.ESC}[6~")


        // Insert
        putCode(TerminalKeyEvent(keyCode = 0x9B), encode = "${ControlCharacters.ESC}[2~")
        // Delete
        putCode(TerminalKeyEvent(keyCode = 0x7F), encode = "${ControlCharacters.ESC}[3~")

        terminal.getTerminalModel().addDataListener(object : DataListener {
            override fun onChanged(key: DataKey<*>, data: Any) {
                this@KeyEncoderImpl.onChanged(key, data)
            }
        })

    }

    override fun encode(event: TerminalKeyEvent): String {
        return mapping[event] ?: nothing
    }

    override fun getTerminal(): Terminal {
        return terminal
    }

    private fun putCode(event: TerminalKeyEvent, encode: String) {
        mapping[event] = encode
    }

    fun arrowKeysApplicationSequences() {
        // Up
        putCode(TerminalKeyEvent(keyCode = 0x26), encode = "${ControlCharacters.ESC}OA")
        // Down
        putCode(TerminalKeyEvent(keyCode = 0x28), encode = "${ControlCharacters.ESC}OB")
        // Left
        putCode(TerminalKeyEvent(keyCode = 0x25), encode = "${ControlCharacters.ESC}OD")
        // Right
        putCode(TerminalKeyEvent(keyCode = 0x27), encode = "${ControlCharacters.ESC}OC")
    }

    fun arrowKeysAnsiCursorSequences() {
        // Up
        putCode(TerminalKeyEvent(keyCode = 0x26), encode = "${ControlCharacters.ESC}[A")
        // Down
        putCode(TerminalKeyEvent(keyCode = 0x28), encode = "${ControlCharacters.ESC}[B")
        // Left
        putCode(TerminalKeyEvent(keyCode = 0x25), encode = "${ControlCharacters.ESC}[D")
        // Right
        putCode(TerminalKeyEvent(keyCode = 0x27), encode = "${ControlCharacters.ESC}[C")
    }


    fun keypadApplicationSequences() {
        // Up
        putCode(TerminalKeyEvent(keyCode = 0xE0), encode = "${ControlCharacters.ESC}OA")
        // Down
        putCode(TerminalKeyEvent(keyCode = 0xE1), encode = "${ControlCharacters.ESC}OB")
        // Left
        putCode(TerminalKeyEvent(keyCode = 0xE2), encode = "${ControlCharacters.ESC}OD")
        // Right
        putCode(TerminalKeyEvent(keyCode = 0xE3), encode = "${ControlCharacters.ESC}OC")
        // Home
        putCode(TerminalKeyEvent(keyCode = 0x24), encode = "${ControlCharacters.ESC}OH")
        // End
        putCode(TerminalKeyEvent(keyCode = 0x23), encode = "${ControlCharacters.ESC}OF")
    }

    fun keypadAnsiSequences() {
        // Up
        putCode(TerminalKeyEvent(keyCode = 0xE0), encode = "${ControlCharacters.ESC}[A")
        // Down
        putCode(TerminalKeyEvent(keyCode = 0xE1), encode = "${ControlCharacters.ESC}[B")
        // Left
        putCode(TerminalKeyEvent(keyCode = 0xE2), encode = "${ControlCharacters.ESC}[D")
        // Right
        putCode(TerminalKeyEvent(keyCode = 0xE3), encode = "${ControlCharacters.ESC}[C")
        // Home
        putCode(TerminalKeyEvent(keyCode = 0x24), encode = "${ControlCharacters.ESC}[H")
        // End
        putCode(TerminalKeyEvent(keyCode = 0x23), encode = "${ControlCharacters.ESC}[F")
    }

    override fun onChanged(key: DataKey<*>, data: Any) {
        if (key == DataKey.ApplicationCursorKeys) {
            if (data as Boolean) {
                arrowKeysApplicationSequences()
            } else {
                arrowKeysAnsiCursorSequences()
            }
        } else if (key == DataKey.AlternateKeypad) {
            if (data as Boolean) {
                keypadApplicationSequences()
            } else {
                keypadAnsiSequences()
            }
        } else if (key == DataKey.AutoNewline) {
            if (data as Boolean) {
                putCode(TerminalKeyEvent(keyCode = 10), encode = "\r\n")
            } else {
                putCode(TerminalKeyEvent(keyCode = 10), encode = "\r")
            }
        }
    }


}