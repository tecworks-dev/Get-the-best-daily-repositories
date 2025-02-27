#include <stdio.h>

#include "arc.h"
#include "kbd_high.h"
#include "si_kbd.h"
#include "ios_usb.h"

#define KBD_BUFFER_SIZE 32

typedef struct _KEYBOARD_BUFFER {
	volatile UCHAR Buffer[KBD_BUFFER_SIZE];
	volatile UCHAR ReadIndex;
	volatile UCHAR WriteIndex;
} KEYBOARD_BUFFER, * PKEYBOARD_BUFFER;


static BYTE s_LastKeycode, s_LastModifier;
static KEYBOARD_BUFFER s_Buffer = { {0}, 0, 0 };
static bool s_CapsLock = false;


// Two lookup tables, encompassing key codes between KEY_A(4) and KEY_SLASH(0x38).
// en_GB keyboard layout, whatever...
//static const UCHAR s_LookupNormal[] = "abcdefghijklmnopqrstuvwxyz1234567890\n\x1b\b\x09 -=[]\\#;'`,./";
//static const UCHAR s_LookupShift [] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!\"£$%^&*()\n\x1b\b\x09 _+{}|~:@~<>?";
static const UCHAR s_LookupNormal[] = "abcdefghijklmnopqrstuvwxyz1234567890\n\x1b\b\x09 -=[]\\\\;#',./";
static const UCHAR s_LookupShift[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!\"£$%^&*()\n\x1b\b\x09 _+{}||:~@<>?";

#define INCREMENT_INDEX(var) (((var) + 1) % sizeof(s_Buffer.Buffer))
#define INCREMENT_INDEX_READ() INCREMENT_INDEX(s_Buffer.ReadIndex)
#define INCREMENT_INDEX_WRITE() INCREMENT_INDEX(s_Buffer.WriteIndex)

static void KBD_Poll(void) {
	SikbdPoll();
	//UlPoll();
	UlkPoll();
}

void KbdHighZeroLast(void) {
	s_LastKeycode = 0;
}

UCHAR IOSKBD_ReadChar() {
	while (s_Buffer.ReadIndex == s_Buffer.WriteIndex) {
		KBD_Poll();
	}
	s_Buffer.ReadIndex = INCREMENT_INDEX_READ();
	return s_Buffer.Buffer[s_Buffer.ReadIndex];
}

bool IOSKBD_CharAvailable() {
	KBD_Poll();
	return s_Buffer.ReadIndex != s_Buffer.WriteIndex;
}

static void KBDWriteChar(UCHAR Character) {
	UCHAR IncWrite = INCREMENT_INDEX_WRITE();
	if (IncWrite != s_Buffer.ReadIndex) {
		s_Buffer.WriteIndex = IncWrite;
		s_Buffer.Buffer[s_Buffer.WriteIndex] = Character;
	}
}

#define KBDWriteString(str) for (int i = 0; i < sizeof(str)-1; i++) KBDWriteChar((str)[i]);

static void KBDWriteKey(BYTE keycode, BYTE modifier) {
	if (keycode < KEY_VALID_START) return;

	if (keycode == KEY_CAPSLOCK) {
		s_CapsLock ^= 1;
		return;
	}
	if (keycode == KEY_SYSRQ) {
		KBDWriteChar('\x80');
		return;
	}
	if (keycode >= KEY_LOOKUP_START && keycode <= KEY_LOOKUP_END) {
		if (s_CapsLock && keycode >= KEY_LOOKUP_START && keycode <= (KEY_LOOKUP_START + 26)) {
			if (modifier & KEY_MODIFIER_SHIFT) modifier &= ~KEY_MODIFIER_SHIFT;
			else modifier |= KEY_MODIFIER_SHIFT;
		}
		if (modifier & KEY_MODIFIER_SHIFT) {
			KBDWriteChar(s_LookupShift[keycode - KEY_LOOKUP_START]);
		}
		else {
			KBDWriteChar(s_LookupNormal[keycode - KEY_LOOKUP_START]);
		}
		return;
	}

	UCHAR ControlChar = 0;

	if (keycode >= KEY_F1 && keycode <= KEY_F12) {
		// This table comes from the ARC specification (ARC/riscspec.pdf) page 105.
		static const char s_ControlChars[] = "PQwxtuqrpMAB";
		KBDWriteChar('\x9b');
		KBDWriteChar('O');
		KBDWriteChar(s_ControlChars[keycode - KEY_F1]);
		return;
	}

	switch (keycode) {
	case KEY_UP:
		ControlChar = 'A';
		break;

	case KEY_DOWN:
		ControlChar = 'B';
		break;

	case KEY_RIGHT:
		ControlChar = 'C';
		break;

	case KEY_LEFT:
		ControlChar = 'D';
		break;

	case KEY_HOME:
		ControlChar = 'H';
		break;

	case KEY_END:
		ControlChar = 'K';
		break;

	case KEY_PAGEUP:
		ControlChar = '?';
		break;

	case KEY_PAGEDOWN:
		ControlChar = '/';
		break;

	case KEY_INSERT:
		ControlChar = '@';
		break;

	case KEY_DELETE:
		ControlChar = 'P';
		break;
	}

	if (ControlChar == 0) return;
	KBDWriteChar('\x9b');
	KBDWriteChar(ControlChar);
}

void KBDOnEvent(PUSB_KBD_REPORT Report) {
	// Only support the first key pressed.
	// Do not check modifier, this prevents things like "press shift, press ;:, release shift first" to input ":;"
	//if (s_LastKeycode == s_Event.report.keycode[0] && s_LastModifier == s_Event.report.modifier) return;
	if (s_LastKeycode == Report->KeyCode[0]) return;
	s_LastKeycode = Report->KeyCode[0];
	s_LastModifier = Report->Modifiers;
	KBDWriteKey(s_LastKeycode, s_LastModifier);
}