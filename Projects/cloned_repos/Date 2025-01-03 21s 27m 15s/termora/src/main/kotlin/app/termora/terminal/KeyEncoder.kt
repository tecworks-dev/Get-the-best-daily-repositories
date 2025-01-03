package app.termora.terminal

/**
 * 对某些特定的键进行编码
 */
interface KeyEncoder {

    /**
     * 终端
     */
    fun getTerminal(): Terminal

    /**
     * 编码
     *
     * @return 返回空字符串表示不处理
     */
    fun encode(event: TerminalKeyEvent): String

}