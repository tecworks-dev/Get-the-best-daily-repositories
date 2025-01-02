package app.termora.terminal

abstract class AbstractProcessor(protected val terminal: Terminal, protected val reader: TerminalReader) : Processor
