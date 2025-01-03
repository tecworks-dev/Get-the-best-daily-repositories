package app.termora.terminal.panel

import app.termora.terminal.TerminalEvent
import app.termora.terminal.TerminalKeyEvent
import app.termora.terminal.TerminalMouseButton
import app.termora.terminal.TerminalMouseEvent
import java.awt.event.InputEvent
import java.awt.event.KeyEvent
import java.awt.event.MouseEvent
import java.awt.event.MouseWheelEvent

private fun awtMouseEvent2TerminalMouseButton(event: MouseEvent): TerminalMouseButton {

    if (event is MouseWheelEvent) {
        return if (event.wheelRotation > 0) TerminalMouseButton.ScrollUp else TerminalMouseButton.ScrollDown
    }

    return when (event.button) {
        1 -> TerminalMouseButton.Left
        2 -> TerminalMouseButton.Middle
        3 -> TerminalMouseButton.Right
        else -> TerminalMouseButton.None
    }
}


private fun getModifierKeys(event: InputEvent): Int {
    var modifier = 0
    if (event.isControlDown) {
        modifier = modifier or TerminalEvent.CTRL_MASK
    }
    if (event.isShiftDown) {
        modifier = modifier or TerminalEvent.SHIFT_MASK
    }
    if (event.isMetaDown) {
        modifier = modifier or TerminalEvent.META_MASK
    }
    if (event.isAltDown) {
        modifier = modifier or TerminalEvent.ALT_MASK
    }
    if (event.isAltGraphDown) {
        modifier = modifier or TerminalEvent.ALT_GRAPH_MASK
    }
    return modifier
}

class AWTTerminalMouseEvent(event: MouseEvent) :
    TerminalMouseEvent(awtMouseEvent2TerminalMouseButton(event), event.clickCount, getModifierKeys(event)) {
}

class AWTTerminalKeyEvent(event: KeyEvent) : TerminalKeyEvent(event.keyCode, getModifierKeys(event))