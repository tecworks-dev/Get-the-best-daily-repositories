package app.termora.terminal

interface Processor {
    fun process(ch: Char): ProcessorState
}