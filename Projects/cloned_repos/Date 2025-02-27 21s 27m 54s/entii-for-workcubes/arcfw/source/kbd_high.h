// High level keyboard API; USB HID keyboard interface.

typedef struct {
	UCHAR Modifiers;
	UCHAR Reserved;
	UCHAR KeyCode[6];
} USB_KBD_REPORT, * PUSB_KBD_REPORT;

enum {
	KEY_MODIFIER_LCTRL = BIT(0),
	KEY_MODIFIER_LSHIFT = BIT(1),
	KEY_MODIFIER_LALT = BIT(2),
	KEY_MODIFIER_LWIN = BIT(3),
	KEY_MODIFIER_RCTRL = BIT(4),
	KEY_MODIFIER_RSHIFT = BIT(5),
	KEY_MODIFIER_RALT = BIT(6),
	KEY_MODIFIER_RWIN = BIT(7),

	KEY_MODIFIER_CTRL = KEY_MODIFIER_LCTRL | KEY_MODIFIER_RCTRL,
	KEY_MODIFIER_SHIFT = KEY_MODIFIER_LSHIFT | KEY_MODIFIER_RSHIFT,
	KEY_MODIFIER_ALT = KEY_MODIFIER_LALT | KEY_MODIFIER_RALT,
	KEY_MODIFIER_WIN = KEY_MODIFIER_LWIN | KEY_MODIFIER_RWIN
};

enum {
	KEY_ERROR_OVF = 1
};

// USB keyboard constants etc, copypasta'd from old Wii ARC firmware
enum {
	KEY_VALID_START = 2,
	KEY_LOOKUP_START = 4,
	KEY_LOOKUP_END = 0x38,

	KEY_CAPSLOCK = 0x39, // Keyboard Caps Lock

	KEY_F1 = 0x3a, // Keyboard F1
	KEY_F2 = 0x3b, // Keyboard F2
	KEY_F3 = 0x3c, // Keyboard F3
	KEY_F4 = 0x3d, // Keyboard F4
	KEY_F5 = 0x3e, // Keyboard F5
	KEY_F6 = 0x3f, // Keyboard F6
	KEY_F7 = 0x40, // Keyboard F7
	KEY_F8 = 0x41, // Keyboard F8
	KEY_F9 = 0x42, // Keyboard F9
	KEY_F10 = 0x43, // Keyboard F10
	KEY_F11 = 0x44, // Keyboard F11
	KEY_F12 = 0x45, // Keyboard F12

	KEY_SYSRQ = 0x46, // Keyboard Print Screen
	KEY_SCROLLLOCK = 0x47, // Keyboard Scroll Lock
	KEY_PAUSE = 0x48, // Keyboard Pause
	KEY_INSERT = 0x49, // Keyboard Insert
	KEY_HOME = 0x4a, // Keyboard Home
	KEY_PAGEUP = 0x4b, // Keyboard Page Up
	KEY_DELETE = 0x4c, // Keyboard Delete Forward
	KEY_END = 0x4d, // Keyboard End
	KEY_PAGEDOWN = 0x4e, // Keyboard Page Down
	KEY_RIGHT = 0x4f, // Keyboard Right Arrow
	KEY_LEFT = 0x50, // Keyboard Left Arrow
	KEY_DOWN = 0x51, // Keyboard Down Arrow
	KEY_UP = 0x52, // Keyboard Up Arrow

	KEY_NUMLOCK = 0x53, // Keyboard Num Lock and Clear
	KEY_KPSLASH = 0x54, // Keypad /
	KEY_KPASTERISK = 0x55, // Keypad *
	KEY_KPMINUS = 0x56, // Keypad -
	KEY_KPPLUS = 0x57, // Keypad +
	KEY_KPENTER = 0x58, // Keypad ENTER
	KEY_KP1 = 0x59, // Keypad 1 and End
	KEY_KP2 = 0x5a, // Keypad 2 and Down Arrow
	KEY_KP3 = 0x5b, // Keypad 3 and PageDn
	KEY_KP4 = 0x5c, // Keypad 4 and Left Arrow
	KEY_KP5 = 0x5d, // Keypad 5
	KEY_KP6 = 0x5e, // Keypad 6 and Right Arrow
	KEY_KP7 = 0x5f, // Keypad 7 and Home
	KEY_KP8 = 0x60, // Keypad 8 and Up Arrow
	KEY_KP9 = 0x61, // Keypad 9 and Page Up
	KEY_KP0 = 0x62, // Keypad 0 and Insert
	KEY_KPDOT = 0x63, // Keypad . and Delete

	KEY_102ND = 0x64, // Keyboard Non-US \ and |
	KEY_COMPOSE = 0x65, // Keyboard Application
	KEY_POWER = 0x66, // Keyboard Power
	KEY_KPEQUAL = 0x67, // Keypad =

	KEY_F13 = 0x68, // Keyboard F13
	KEY_F14 = 0x69, // Keyboard F14
	KEY_F15 = 0x6a, // Keyboard F15
	KEY_F16 = 0x6b, // Keyboard F16
	KEY_F17 = 0x6c, // Keyboard F17
	KEY_F18 = 0x6d, // Keyboard F18
	KEY_F19 = 0x6e, // Keyboard F19
	KEY_F20 = 0x6f, // Keyboard F20
	KEY_F21 = 0x70, // Keyboard F21
	KEY_F22 = 0x71, // Keyboard F22
	KEY_F23 = 0x72, // Keyboard F23
	KEY_F24 = 0x73, // Keyboard F24

	KEY_OPEN = 0x74, // Keyboard Execute
	KEY_HELP = 0x75, // Keyboard Help
	KEY_PROPS = 0x76, // Keyboard Menu
	KEY_FRONT = 0x77, // Keyboard Select
	KEY_STOP = 0x78, // Keyboard Stop
	KEY_AGAIN = 0x79, // Keyboard Again
	KEY_UNDO = 0x7a, // Keyboard Undo
	KEY_CUT = 0x7b, // Keyboard Cut
	KEY_COPY = 0x7c, // Keyboard Copy
	KEY_PASTE = 0x7d, // Keyboard Paste
	KEY_FIND = 0x7e, // Keyboard Find
	KEY_MUTE = 0x7f, // Keyboard Mute
	KEY_VOLUMEUP = 0x80, // Keyboard Volume Up
	KEY_VOLUMEDOWN = 0x81, // Keyboard Volume Down
	// = 0x82  Keyboard Locking Caps Lock
	// = 0x83  Keyboard Locking Num Lock
	// = 0x84  Keyboard Locking Scroll Lock
	KEY_KPCOMMA = 0x85, // Keypad Comma
	// = 0x86  Keypad Equal Sign
	KEY_RO = 0x87, // Keyboard International1
	KEY_KATAKANAHIRAGANA = 0x88, // Keyboard International2
	KEY_YEN = 0x89, // Keyboard International3
	KEY_HENKAN = 0x8a, // Keyboard International4
	KEY_MUHENKAN = 0x8b, // Keyboard International5
	KEY_KPJPCOMMA = 0x8c, // Keyboard International6
	// = 0x8d  Keyboard International7
	// = 0x8e  Keyboard International8
	// = 0x8f  Keyboard International9
	KEY_HANGEUL = 0x90, // Keyboard LANG1
	KEY_HANJA = 0x91, // Keyboard LANG2
	KEY_KATAKANA = 0x92, // Keyboard LANG3
	KEY_HIRAGANA = 0x93, // Keyboard LANG4
	KEY_ZENKAKUHANKAKU = 0x94, // Keyboard LANG5
	// = 0x95  Keyboard LANG6
	// = 0x96  Keyboard LANG7
	// = 0x97  Keyboard LANG8
	// = 0x98  Keyboard LANG9
	// = 0x99  Keyboard Alternate Erase
	// = 0x9a  Keyboard SysReq/Attention
	// = 0x9b  Keyboard Cancel
	// = 0x9c  Keyboard Clear
	// = 0x9d  Keyboard Prior
	// = 0x9e  Keyboard Return
	// = 0x9f  Keyboard Separator
	// = 0xa0  Keyboard Out
	// = 0xa1  Keyboard Oper
	// = 0xa2  Keyboard Clear/Again
	// = 0xa3  Keyboard CrSel/Props
	// = 0xa4  Keyboard ExSel

	// = 0xb0  Keypad 00
	// = 0xb1  Keypad 000
	// = 0xb2  Thousands Separator
	// = 0xb3  Decimal Separator
	// = 0xb4  Currency Unit
	// = 0xb5  Currency Sub-unit
	KEY_KPLEFTPAREN = 0xb6, // Keypad (
	KEY_KPRIGHTPAREN = 0xb7, // Keypad )
	// = 0xb8  Keypad {
	// = 0xb9  Keypad }
	// = 0xba  Keypad Tab
	// = 0xbb  Keypad Backspace
	// = 0xbc  Keypad A
	// = 0xbd  Keypad B
	// = 0xbe  Keypad C
	// = 0xbf  Keypad D
	// = 0xc0  Keypad E
	// = 0xc1  Keypad F
	// = 0xc2  Keypad XOR
	// = 0xc3  Keypad ^
	// = 0xc4  Keypad %
	// = 0xc5  Keypad <
	// = 0xc6  Keypad >
	// = 0xc7  Keypad &
	// = 0xc8  Keypad &&
	// = 0xc9  Keypad |
	// = 0xca  Keypad ||
	// = 0xcb  Keypad :
	// = 0xcc  Keypad #
	// = 0xcd  Keypad Space
	// = 0xce  Keypad @
	// = 0xcf  Keypad !
	// = 0xd0  Keypad Memory Store
	// = 0xd1  Keypad Memory Recall
	// = 0xd2  Keypad Memory Clear
	// = 0xd3  Keypad Memory Add
	// = 0xd4  Keypad Memory Subtract
	// = 0xd5  Keypad Memory Multiply
	// = 0xd6  Keypad Memory Divide
	// = 0xd7  Keypad +/-
	// = 0xd8  Keypad Clear
	// = 0xd9  Keypad Clear Entry
	// = 0xda  Keypad Binary
	// = 0xdb  Keypad Octal
	// = 0xdc  Keypad Decimal
	// = 0xdd  Keypad Hexadecimal

	KEY_LEFTCTRL = 0xe0, // Keyboard Left Control
	KEY_LEFTSHIFT = 0xe1, // Keyboard Left Shift
	KEY_LEFTALT = 0xe2, // Keyboard Left Alt
	KEY_LEFTMETA = 0xe3, // Keyboard Left GUI
	KEY_RIGHTCTRL = 0xe4, // Keyboard Right Control
	KEY_RIGHTSHIFT = 0xe5, // Keyboard Right Shift
	KEY_RIGHTALT = 0xe6, // Keyboard Right Alt
	KEY_RIGHTMETA = 0xe7, // Keyboard Right GUI
};



UCHAR IOSKBD_ReadChar();
bool IOSKBD_CharAvailable();
void KBDOnEvent(PUSB_KBD_REPORT Report);
void KbdHighZeroLast(void);