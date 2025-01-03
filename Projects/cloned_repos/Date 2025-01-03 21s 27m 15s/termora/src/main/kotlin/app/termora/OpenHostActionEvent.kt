package app.termora

import java.awt.event.ActionEvent

class OpenHostActionEvent(source: Any, val host: Host) : ActionEvent(source, ACTION_PERFORMED, String())