package app.termora.terminal

interface ProcessorState {
}

internal enum class TerminalState : ProcessorState {
    /**
     * 就绪状态
     */
    READY,

    /**
     * ESC
     */
    EscapeSequence,

    /**
     * Control Sequence Introducer (CSI  is 0x9b).
     */
    CSI,

    /**
     * Operating System Command (OSC  is 0x9d).
     */
    OSC,

    /**
     * https://invisible-island.net/xterm/ctlseqs/ctlseqs.html#h4-Controls-beginning-with-ESC:ESC-lparen-C.F20
     */
    ESC_LPAREN,


    /**
     * Device Control String (DCS  is 0x90).
     */
    DCS,

    /**
     * 处理普通文本
     */
    Text,
;

    override fun toString(): String {
        return name
    }
}