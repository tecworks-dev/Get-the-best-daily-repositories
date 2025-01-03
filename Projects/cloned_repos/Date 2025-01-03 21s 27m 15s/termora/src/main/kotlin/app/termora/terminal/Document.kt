package app.termora.terminal

interface Document {

    /**
     * 获取整个文档的所有内容
     */
    fun getText(): String

    /**
     * @param n 0:清除从光标位置到屏幕末尾的部分 ; 1:清除从光标位置到屏幕开头的部分 ; 2:清除整个屏幕（在DOS ANSI.SYS中，光标还会向左上方移动）; 3: 清除整个屏幕，并删除回滚缓存区中的所有行（这个特性是xterm添加的，其他终端应用程序也支持）
     */
    fun eraseInDisplay(n: Int)

    /**
     * @param n 0:清除从光标位置到该行末尾的部分 ; 1: 清除从光标位置到该行开头的部分 ; 2: 清除整行。光标位置不变
     */
    fun eraseInLine(n: Int)

    /**
     * 获取某一行
     * @param row 下标从 1 开始
     */
    fun getLine(row: Int): TerminalLine

    /**
     * 获取可视区域行
     */
    fun getScreenLine(row: Int): TerminalLine

    /**
     * 写入
     */
    fun write(text: String)

    /**
     * 获取总行数，包括历史行数
     */
    fun getLineCount(): Int

    /**
     * 缓冲区
     */
    fun getCurrentTerminalLineBuffer(): TerminalLineBuffer

    /**
     * @see [DataKey.AlternateScreenBuffer] 模式下的
     */
    fun getScreenTerminalLineBuffer(): TerminalLineBuffer

    /**
     * 常规模式下的
     */
    fun getTerminalLineBuffer(): TerminalLineBuffer

    /**
     * 并非滚动条，而是数据滚动。当 [TerminalModel.isAlternateScreenBuffer] & [DataKey.ScrollingRegion] 的时候生效
     *
     * @param button 仅支持 [TerminalMouseButton.ScrollUp] or [TerminalMouseButton.ScrollDown]
     */

    fun scroll(button: TerminalMouseButton, count: Int = 1)

    /**
     * 添加新行，如果 [DataKey.AutoNewline] 为 false 那么光标移动到下一行
     */
    fun newline()

    /**
     * 获取终端信息
     */
    fun getTerminal(): Terminal

    /**
     * 删除行，从 [offset] 开始往下删除 [count] 行
     *
     * @param offset 下标从 0 开始
     * @param count 删除多少个
     */
    fun deleteLines(offset: Int, count: Int)

}

