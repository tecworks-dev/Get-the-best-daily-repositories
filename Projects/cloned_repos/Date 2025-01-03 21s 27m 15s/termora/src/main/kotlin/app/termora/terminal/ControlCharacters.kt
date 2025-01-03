package app.termora.terminal


interface ControlCharacters {
    companion object {

        /**
         * \a
         */
        const val BEL = 0x07.toChar()

        /**
         * \b
         */
        const val BS = 0x08.toChar()

        /**
         * https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h4-Single-character-functions:SI.BE1
         */
        const val SI = '\u000F'

        /**
         * https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h4-Single-character-functions:SO.BE7
         */
        const val SO = '\u000E'

        /**
         * \r
         */
        const val CR = 0x0D.toChar()

        const val ENQ = 0x05.toChar()

        /**
         * \f
         */
        const val FF = 0x0C.toChar()

        /**
         * \n
         */
        const val LF = 0x0A.toChar()

        /**
         * SPACE
         */
        const val SP = ' '

        /**
         * \v
         */
        const val VT = 0x0B.toChar()


        /**
         * \t
         */
        const val TAB = 0x09.toChar()

        /**
         * \
         */
        const val ST = 0x9c.toChar()


        const val ESC = 0x1B.toChar()

    }
}