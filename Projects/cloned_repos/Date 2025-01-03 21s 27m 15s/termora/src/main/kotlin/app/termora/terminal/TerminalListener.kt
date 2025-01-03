package app.termora.terminal


interface TerminalListener {

    /**
     * 关闭后回调，收到这个方法之后应该执行资源回收
     */
    fun onClose(terminal: Terminal) {}

}