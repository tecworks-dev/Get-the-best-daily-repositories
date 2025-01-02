package app.termora.terminal

import org.slf4j.LoggerFactory

class DeviceControlProcessor(private val terminal: Terminal) : Processor {
    private val args = StringBuilder()

    companion object {
        private val log = LoggerFactory.getLogger(DeviceControlProcessor::class.java)
    }


    override fun process(ch: Char): ProcessorState {
        val state = when (ch) {
            ControlCharacters.ST -> {
                if (log.isWarnEnabled) {
                    log.warn("Ignore DCS: {}", args)
                }
                TerminalState.READY
            }

            else -> {
                args.append(ch)
                TerminalState.DCS
            }
        }

        if (state == TerminalState.READY) {
            args.clear()
        }

        return state

    }

}