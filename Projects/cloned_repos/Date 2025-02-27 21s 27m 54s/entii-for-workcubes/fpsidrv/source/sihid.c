// SI high-level
#define DEVL 1
#include <ntddk.h>
#include <arc.h> // LOADER_PARAMETER_BLOCK
#include <stdio.h>
#include "runtime.h"
#include "si.h"
#include "keyboard.h"
#include "mouse.h"

#define ARC_BIT(x) BIT(x)

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

enum {
	SI_CMD_RESET = 0xFF,
	SI_CMD_READ_GC = 0x40,
	SI_CMD_READ_GCKBD = 0x54,
	SI_CMD_READ_64 = 0x01,
	SI_CMD_READ_64KBD = 0x13
};

enum {
	SI_CONTROLLER_TYPE_N64 = 0,
	SI_CONTROLLER_TYPE_GC = 1
};

// N64 enums/tables

enum {
	SI_N64_TYPE_CONTROLLER = 0x500,
	SI_N64_TYPE_KEYBOARD = 0x002,
	SI_N64_TYPE_MOUSE = 0x200,
};

enum {
	N64PAD_CRIGHT = ARC_BIT(0),
	N64PAD_CLEFT = ARC_BIT(1),
	N64PAD_CDOWN = ARC_BIT(2),
	N64PAD_CUP = ARC_BIT(3),
	N64PAD_R = ARC_BIT(4),
	N64PAD_L = ARC_BIT(5),

	N64PAD_RIGHT = ARC_BIT(8),
	N64PAD_LEFT = ARC_BIT(9),
	N64PAD_DOWN = ARC_BIT(10),
	N64PAD_UP = ARC_BIT(11),
	N64PAD_START = ARC_BIT(12),
	N64PAD_Z = ARC_BIT(13),
	N64PAD_B = ARC_BIT(14),
	N64PAD_A = ARC_BIT(15),
	
	N64MOUSE_LEFT = N64PAD_A,
	N64MOUSE_RIGHT = N64PAD_B
};

enum {
	KBD64_LED_NUM = ARC_BIT(0),
	KBD64_LED_CAPS = ARC_BIT(1),
	KBD64_LED_POWER = ARC_BIT(2)
};

enum {
	USB_MOUSE_LEFT = ARC_BIT(0),
	USB_MOUSE_RIGHT = ARC_BIT(1),
	USB_MOUSE_CENTRE = ARC_BIT(2),
};

static const UCHAR sc_64kbdToUsbTable1[] = {
	0x03, // 0201 none
	0x03, // 0301 none
	0x03, // 0401 none
	0x1a, // 0501 w
	0x08, // 0601 e
	0x15, // 0701 r
	0x17, // 0801 t
	0x1c, // 0901 y
	0x3b, // 0a01 [F2]
	0x3a, // 0b01 [F1]
	0x14, // 0c01 q
	0x2b, // 0d01 [TAB]
	0xe1, // 0e01 [left shift]
};

static const UCHAR sc_64kbdToUsbTable2[] = {
	0x03, // 0202 none
	0x03, // 0302 none
	0x03, // 0402 none
	0x03, // 0502 none
	0x2c, // 0602 [space]
	0x38, // 0702 /?
	0x37, // 0802 .>
	0x36, // 0902 ,<
	0x3f, // 0a02 [F6]
	0x40, // 0b02 [F7]
	0x03, // 0c02 none
	0x03, // 0d02 none
	0xe7, // 0e02 [right windows]
	0x03, // 0f02 none
	0xe3, // 1002 [left windows]
};

static const UCHAR sc_64kbdToUsbTable3[] = {
	0x44, // 0203 F11
	0x03, // 0303 none
	0x03, // 0403 none
	0x03, // 0503 none
	0x34, // 0603 '@
	0x33, // 0703 ;:
	0x0f, // 0803 l
	0x0e, // 0903 k
	0x41, // 0a03 [F8]
	0x42, // 0b03 [F9]
};

static const UCHAR sc_64kbdToUsbTable4[] = {
	0x52, // 0204 [up]
	0x03, // 0304 none
	0x03, // 0404 none
	0x55, // 0504 [numpad *]
	0x13, // 0604 p
	0x12, // 0704 o
	0x0c, // 0804 i
	0x18, // 0904 u
	0x43, // 0a04 [F10]
	0x03, // 0b04 none
	0x2f, // 0c04 [{
	0x28, // 0d04 [ENTER]
	0x03, // 0e04 none
	0x03, // 0f04 none
	0x31, // 1004 \|
};

static const UCHAR sc_64kbdToUsbTable5[] = {
	0x50, // 0205 [left]
	0x51, // 0305 [down]
	0x4f, // 0405 [right]
	0x1f, // 0505 2"
	0x20, // 0605 3£
	0x21, // 0705 4$
	0x22, // 0805 5%
	0x23, // 0905 6^
	0x53, // 0a05 [num lock]
	0x00, // 0b05 [menu] - (next to next/previous page) - don't map to anything for now
	0x1e, // 0c05 1!
	0x35, // 0d05 `¬
	0x03, // 0e05 none
	0x39, // 0f05 [caps lock]
	0x03, // 1005 none
	0x31, // 1105 \|
};

static const UCHAR sc_64kbdToUsbTable6[] = {
	0x4d, // 0206 [end]
	0x03, // 0306 none
	0x30, // 0406 ]}
	0x2d, // 0506 -_
	0x27, // 0606 0)
	0x26, // 0706 9(
	0x25, // 0806 8*
	0x24, // 0906 7&
	0x03, // 0a06 none
	0x45, // 0b06 [F12]
	0x2e, // 0c06 =+
	0x2a, // 0d06 [backspace]
	0xe5, // 0e06 [right shift]
	0x03, // 0f06 none
	0x65, // 1006 [context menu]
};

static const UCHAR sc_64kbdToUsbTable7[] = {
	0x00, // 0207 [next page] - don't map to anything for now
	0x03, // 0307 none
	0x03, // 0407 none
	0x07, // 0507 d
	0x09, // 0607 f
	0x0a, // 0707 g
	0x0b, // 0807 h
	0x0d, // 0907 j
	0x3d, // 0a07 [F4]
	0x3e, // 0b07 [F5]
	0x16, // 0c07 s
	0x04, // 0d07 a
	0x03, // 0e07 none
	0x03, // 0f07 none
	0x03, // 1007 none
	0xe0, // 1107 [left ctrl]
};

static const UCHAR sc_64kbdToUsbTable8[] = {
	0x00, // 0208 [previous page] - don't map to anything for now
	0x03, // 0308 none
	0x03, // 0408 none
	0x06, // 0508 c
	0x19, // 0608 v
	0x05, // 0708 b
	0x11, // 0808 n
	0x10, // 0908 m
	0x29, // 0a08 [ESC]
	0x3c, // 0b08 [F3]
	0x1b, // 0c08 x
	0x1d, // 0d08 z
	0x03, // 0e08 none
	0x03, // 0f08 none
	0xe2, // 1008 [left alt]
};

// GC enums/tables

enum {
	SI_GC_TYPE_KEYBOARD = 0x020,
	SI_GC_TYPE_KEYBOARD2 = 0x030 // PSO does support this
};

enum {
	GCPAD_LEFT = ARC_BIT(0),
	GCPAD_RIGHT = ARC_BIT(1),
	GCPAD_DOWN = ARC_BIT(2),
	GCPAD_UP = ARC_BIT(3),
	GCPAD_Z = ARC_BIT(4),
	GCPAD_R = ARC_BIT(5),
	GCPAD_L = ARC_BIT(6),
	GCPAD_A = ARC_BIT(8),
	GCPAD_B = ARC_BIT(9),
	GCPAD_X = ARC_BIT(10),
	GCPAD_Y = ARC_BIT(11),
	GCPAD_START = ARC_BIT(12)
};

enum { // in low nibble of type. PSO supports all of these and has key tables for all of them
	GC_LAYOUT_JAPAN, // JIS
	GC_LAYOUT_US, // qwerty
	GC_LAYOUT_FR, // azerty
	GC_LAYOUT_DE, // qwertz
	GC_LAYOUT_ES, // qwerty(es-ES)
};

enum {
	SI_GC_SCANCODE_LOOKUP_START = 0x10,
	SI_GC_SCANCODE_LOOKUP_END = 0x34,
	SI_GC_SCANCODE_LOOKUP_OFFSET = (SI_GC_SCANCODE_LOOKUP_START - KEY_LOOKUP_START),

	SI_GC_SCANCODE_LOOKUP_START2 = 0x40,
	SI_GC_SCANCODE_LOOKUP_END2 = 0x4C,
	SI_GC_SCANCODE_LOOKUP_OFFSET2 = (SI_GC_SCANCODE_LOOKUP_START2 - KEY_F1),

	SI_GC_SCANCODE_END = 0x62
};

enum {
	GC_KEY_LCTRL = 0x56,
	GC_KEY_LSHIFT = 0x54,
	GC_KEY_LALT = 0x57,
	GC_KEY_LWIN = 0x58,
	GC_KEY_RSHIFT = 0x55,
	GC_KEY_RWIN = 0x5a,
	GC_KEY_CAPSLOCK = 0x53,
};

static const UCHAR sc_GckbdToUsbTable0[] =
{
	0x00, // 0  No key pressed
	0x01, // 1  Error
	0x02, // 2  Error
	0x03, // 3  Unused
	0x03, // 4  Unused
	0x03, // 5  Unused
	0x4a, // 6  [home]
	0x4d, // 7  [end]
	0x4b, // 8  [pageup]
	0x4e, // 9  [pagedown]
	0x47, // a  [scroll lock]
	0x03, // b  Unused
	0x03, // c  Unused
	0x03, // d  Unused
	0x03, // e  Unused
	0x03, // f  Unused
};
_Static_assert(sizeof(sc_GckbdToUsbTable0) == SI_GC_SCANCODE_LOOKUP_START);

static const UCHAR sc_GckbdToUsbTable34[] =
{
	0x2d, // 34 -_
	0x2e, // 35 =+
	0x31, // 36 \|
	0x55, // 37 [numpad *]
	0x2f, // 38 [{
	0x33, // 39 ;:
	0x34, // 3a '@
	0x30, // 3b ]}
	0x36, // 3c ,<
	0x37, // 3d .>
	0x38, // 3e /?
	0x31, // 3f \|
};
_Static_assert(sizeof(sc_GckbdToUsbTable34) == (SI_GC_SCANCODE_LOOKUP_START2 - SI_GC_SCANCODE_LOOKUP_END));

static const UCHAR sc_GckbdToUsbTable4C[] =
{
	0x29, // 4c [ESC]
	0x49, // 4d [Insert]
	0x4c, // 4e [Delete]
	0x35, // 4f `¬
	0x2a, // 50 [backspace]
	0x2b, // 51 [TAB]
	0x03, // 52 Unused
	0x39, // 53 [Caps Lock]
	0xe1, // 54 [Left Shift]
	0xe5, // 55 [Right Shift]
	0xe0, // 56 [Left Ctrl]
	0xe2, // 57 [Left Alt]
	0xe3, // 58 [Left Windows]
	0x2c, // 59 [space]
	0xe7, // 5a [Right Windows]
	0x65, // 5b [context menu]
	0x50, // 5c [left]
	0x51, // 5d [down]
	0x52, // 5e [up]
	0x4f, // 5f [right]
	0x03, // 60 Unused
	0x28, // 61 [ENTER]
};
_Static_assert(sizeof(sc_GckbdToUsbTable4C) == (SI_GC_SCANCODE_END - SI_GC_SCANCODE_LOOKUP_END2));

static JOYBUS_DEVICE_TYPE s_ConnectedDevices[SI_CHANNEL_COUNT];

static USB_KBD_REPORT s_Report = { 0 };
static USB_MOUSE_REPORT s_MouseReport = { 0 };
static BOOLEAN s_CapsLock = FALSE;
static BOOLEAN s_NumLock = FALSE;
static BOOLEAN s_InTextSetup = FALSE;

static BOOLEAN SikbdpUsbAddKey(PUSB_KBD_REPORT Report, UCHAR UsbScanCode) {
	UCHAR FreeSlot = sizeof(Report->KeyCode);
	for (UCHAR i = 0; i < sizeof(Report->KeyCode); i++) {
		if (Report->KeyCode[i] == UsbScanCode) return TRUE;
		if (Report->KeyCode[i] == 0 && FreeSlot >= sizeof(Report->KeyCode))
			FreeSlot = i;
	}

	if (FreeSlot >= sizeof(Report->KeyCode)) return FALSE;
	Report->KeyCode[FreeSlot] = UsbScanCode;
	return TRUE;
}

static BOOLEAN SikbdpUsbRemoveKey(PUSB_KBD_REPORT Report, UCHAR UsbScanCode) {
	BOOLEAN Changed = FALSE;
	for (UCHAR i = 0; i < sizeof(Report->KeyCode); i++) {
		if (Report->KeyCode[i] != UsbScanCode) continue;
		Report->KeyCode[i] = 0;
		Changed = TRUE;
	}
	return Changed;
}

static BOOLEAN SikbdpUpdateKey(PUSB_KBD_REPORT Report, UCHAR UsbScanCode, BOOLEAN Released) {
	if (UsbScanCode <= 3) return FALSE;
	if (Released) return SikbdpUsbRemoveKey(Report, UsbScanCode);
	return SikbdpUsbAddKey(Report, UsbScanCode);
}

static BOOLEAN SikbdpUsbUpdateModifier(PUSB_KBD_REPORT Report, UCHAR UsbModifierBit, BOOLEAN Released) {
	// check if the bit is already correct
	BOOLEAN IsSet = (Report->Modifiers & UsbModifierBit) != 0;
	if (IsSet == !Released) return FALSE;

	// bit is incorrect, so flip it
	Report->Modifiers ^= UsbModifierBit;
	return TRUE;
}

static void SikbdpReset(ULONG channel) {
	// Send reset command to device, use the response value as the actual device type.
	JOYBUS_DEVICE_TYPE DeviceType;
	DeviceType.Value = 0;
	if (!NT_SUCCESS(SiTransferByteSync(channel, SI_CMD_RESET, &DeviceType, 3))) {
		s_ConnectedDevices[channel].Value = 0xFFFFFFFF;
		return;
	}
	s_ConnectedDevices[channel] = DeviceType;
}

static BOOLEAN SikbdpInitPolling(ULONG channel);

static BOOLEAN SikbdpSetupddLoaded(void) {
	// Determine if setupdd is loaded.
	// We do this by checking if KeLoaderBlock->SetupLoaderBlock is non-NULL.
	// This is the same way that kernel itself does it, and offset of this elem is stable.
	PLOADER_PARAMETER_BLOCK LoaderBlock = *(PLOADER_PARAMETER_BLOCK*)KeLoaderBlock;
	return LoaderBlock->SetupLoaderBlock != NULL;
}

BOOLEAN SikbdInit(void) {
	s_InTextSetup = SikbdpSetupddLoaded();
	SiTogglePoll(FALSE);
	// Get device type for each connected device.
	for (ULONG channel = 0; channel < SI_CHANNEL_COUNT; channel++)
		s_ConnectedDevices[channel] = SiGetDeviceTypeReset(channel);

	for (ULONG channel = 0; channel < SI_CHANNEL_COUNT; channel++) {
		if (!SiDeviceTypeValid(s_ConnectedDevices[channel])) continue;

		SikbdpReset(channel);
	}
	
	BOOLEAN EnablePolling = FALSE;
	for (ULONG channel = 0; channel < SI_CHANNEL_COUNT; channel++) {
		if (SikbdpInitPolling(channel)) EnablePolling = TRUE;
	}
	
	if (EnablePolling) SiTogglePoll(TRUE);

	// All done.
	return EnablePolling;
}

static BOOLEAN Sikbdp64UpdateModifiers(PUSB_KBD_REPORT Report, UCHAR ScanCode, BOOLEAN Released) {
	switch (ScanCode) {
	case KEY_LEFTCTRL:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_LCTRL, Released);
	case KEY_LEFTSHIFT:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_LSHIFT, Released);
	case KEY_LEFTALT:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_LALT, Released);
	case KEY_LEFTMETA:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_LWIN, Released);
	case KEY_RIGHTSHIFT:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_RSHIFT, Released);
	case KEY_RIGHTMETA:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_RWIN, Released);
	default:
		return FALSE;
	}
}

static BOOLEAN Sikbdp64KeyIsModifier(UCHAR UsbScanCode) {
	switch (UsbScanCode) {
	case KEY_LEFTCTRL:
	case KEY_LEFTSHIFT:
	case KEY_LEFTALT:
	case KEY_LEFTMETA:
	case KEY_RIGHTSHIFT:
	case KEY_RIGHTMETA:
		return TRUE;
	default:
		return FALSE;
	}
	return FALSE;
}

static UCHAR Sikbdp64ConvertScancode(USHORT data) {
	// convert scancode to USB HID scancode
	if (data == 0) return 0; // key not pressed

	UCHAR data0 = (UCHAR) (data >> 8);
	UCHAR data1 = (UCHAR) data;

	if (data1 == 0 || data1 > 8 || data0 < 2) return 3; // error

	data0 -= 2;

#define KBD64_CASE(x) case x :\
	if (data0 >= sizeof(sc_64kbdToUsbTable##x)) return 3; /* error */ \
	return sc_64kbdToUsbTable##x [data0]

	switch (data1) {
		KBD64_CASE(1);
		KBD64_CASE(2);
		KBD64_CASE(3);
		KBD64_CASE(4);
		KBD64_CASE(5);
		KBD64_CASE(6);
		KBD64_CASE(7);
		KBD64_CASE(8);
	}

#undef KBD64_CASE

	// should not get here
	return 3;
}
static BOOLEAN SikbdpConvert64Usb(PUSB_KBD_REPORT Report, PUSHORT Data, UCHAR Offset, UCHAR Length, UCHAR Flags) {
	if ((Flags & ARC_BIT(4))) return FALSE; // Error flag set, leave keyboard data untouched.

	BOOLEAN Changed = FALSE;

	// Home key is bit 0 of the flags instead of its own scancode for some reason.
	if (Flags & ARC_BIT(0)) {
		Changed = SikbdpUpdateKey(Report, KEY_HOME, FALSE);
	}

	for (UCHAR i = Offset; i < Length; i++) {
		UCHAR ScanCode = Sikbdp64ConvertScancode(Data[i]);
		if (ScanCode <= 3) continue;
		if (ScanCode == KEY_CAPSLOCK) s_CapsLock ^= 1;
		if (ScanCode == KEY_NUMLOCK) s_NumLock ^= 1;
		BOOLEAN ChangedThis = FALSE;
		if (Sikbdp64KeyIsModifier(ScanCode)) {
			ChangedThis = Sikbdp64UpdateModifiers(Report, ScanCode, FALSE);
		}
		else {
			// Not a modifier
			ChangedThis = SikbdpUpdateKey(Report, ScanCode, FALSE);
		}
		Changed = Changed || ChangedThis;
	}
	return Changed;
}

static void Sikbdp64ReceivePart(PUSHORT Data, UCHAR Offset, UCHAR Length, UCHAR Flags) {
	// ARC only likes one key-input down at a time:
	memset(s_Report.KeyCode, 0, sizeof(s_Report.KeyCode));
	s_Report.Modifiers = 0;
	BOOLEAN Changed = SikbdpConvert64Usb(&s_Report, Data, Offset, Length, Flags);
	KbdReadComplete(&s_Report);
}

static BOOLEAN SikbdpGcKeyIsModifier(UCHAR GcScanCode) {
	switch (GcScanCode) {
	case GC_KEY_LCTRL:
	case GC_KEY_LSHIFT:
	case GC_KEY_LALT:
	case GC_KEY_LWIN:
	case GC_KEY_RSHIFT:
	case GC_KEY_RWIN:
		return TRUE;
	default:
		return FALSE;
	}
	return FALSE;
}

static BOOLEAN SikbdpGcUpdateModifiers(PUSB_KBD_REPORT Report, UCHAR GcScanCode, BOOLEAN Released) {
	switch (GcScanCode) {
	case GC_KEY_LCTRL:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_LCTRL, Released);
	case GC_KEY_LSHIFT:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_LSHIFT, Released);
	case GC_KEY_LALT:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_LALT, Released);
	case GC_KEY_LWIN:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_LWIN, Released);
	case GC_KEY_RSHIFT:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_RSHIFT, Released);
	case GC_KEY_RWIN:
		return SikbdpUsbUpdateModifier(Report, KEY_MODIFIER_RWIN, Released);
	default:
		return FALSE;
	}
}

static UCHAR SikbdpGcConvertScancode(UCHAR data) {
	// convert scancode to USB HID scancode
	if (data >= SI_GC_SCANCODE_LOOKUP_START && data < SI_GC_SCANCODE_LOOKUP_END) {
		data -= SI_GC_SCANCODE_LOOKUP_OFFSET;
	}
	else if (data >= SI_GC_SCANCODE_LOOKUP_START2 && data < SI_GC_SCANCODE_LOOKUP_END2) {
		data -= SI_GC_SCANCODE_LOOKUP_OFFSET2;
	}
	else if (data < SI_GC_SCANCODE_LOOKUP_START) {
		data = sc_GckbdToUsbTable0[data];
	}
	else if (data < SI_GC_SCANCODE_LOOKUP_START2) {
		data = sc_GckbdToUsbTable34[data - SI_GC_SCANCODE_LOOKUP_END];
	}
	else {
		data = sc_GckbdToUsbTable4C[data - SI_GC_SCANCODE_LOOKUP_END2];
	}
	return data;
}

static BOOLEAN SikbdpConvertGcUsb(PUSB_KBD_REPORT Report, PUCHAR Data, UCHAR Offset, UCHAR Length) {
	BOOLEAN Changed = FALSE;
	for (UCHAR i = Offset; i < Length; i++) {
		if (Data[i] < 3 || Data[i] >= SI_GC_SCANCODE_END) continue;
		UCHAR ScanCode = Data[i];
		BOOLEAN ChangedThis = FALSE;
		if (SikbdpGcKeyIsModifier(ScanCode)) {
			ChangedThis = SikbdpGcUpdateModifiers(Report, ScanCode, FALSE);
		}
		else {
			// Not a modifier
			ChangedThis = SikbdpUpdateKey(Report, SikbdpGcConvertScancode(ScanCode), FALSE);
		}
		Changed = Changed || ChangedThis;
	}
	return Changed;
}

static void SikbdpGcReceivePart(PUCHAR Data, UCHAR Offset, UCHAR Length) {
	// ARC only likes one key-input down at a time:
	memset(s_Report.KeyCode, 0, sizeof(s_Report.KeyCode));
	s_Report.Modifiers = 0;
	BOOLEAN Changed = SikbdpConvertGcUsb(&s_Report, Data, Offset, Length);
	KbdReadComplete(&s_Report);
}

static void SikbdpPollCallbackGcKbd(ULONG Channel, PUCHAR Data) {
	// GC keyboard scancodes at bytes 4-7
	SikbdpGcReceivePart(&Data[4], 0, 3);
}

enum {
	CONTROLLER_SCANCODE_FRAME_COUNT = 10
};

static void SikbdpPollCallbackGcCon(ULONG Channel, PUCHAR Data) {
	// 16 bit buttons ; 16 bit analog stick (8 bit x/y) ; 32 bit analog triggers (8 bit x/y/l/r)
	USHORT Buttons = ((USHORT)Data[0] << 8) | Data[1];

	signed char StickY = (char)((UCHAR)(Data[3]) - 0x80);
	signed char StickX = (char)((UCHAR)(Data[2]) - 0x80);
	signed char CStickY = (char)((UCHAR)(Data[5]) - 0x80);
	signed char CStickX = (char)((UCHAR)(Data[4]) - 0x80);
	
	if (s_InTextSetup) {
		// Convert to keyboard scancodes for text setup.
		UCHAR KbdScan[10] = { 0 };
		UCHAR KbdIndex = 0;

		if (KbdIndex < sizeof(KbdScan) && StickX > 64) KbdScan[KbdIndex++] = 0x5f; // right
		if (KbdIndex < sizeof(KbdScan) && StickX < -64) KbdScan[KbdIndex++] = 0x5c; // left
		if (KbdIndex < sizeof(KbdScan) && StickY > 64) KbdScan[KbdIndex++] = 0x5e; // up
		if (KbdIndex < sizeof(KbdScan) && StickY < -64) KbdScan[KbdIndex++] = 0x5d; // down

		if (KbdIndex < sizeof(KbdScan) && (Buttons & GCPAD_RIGHT) != 0) KbdScan[KbdIndex++] = 0x5f; // right
		if (KbdIndex < sizeof(KbdScan) && (Buttons & GCPAD_LEFT) != 0) KbdScan[KbdIndex++] = 0x5c; // left
		if (KbdIndex < sizeof(KbdScan) && (Buttons & GCPAD_UP) != 0) KbdScan[KbdIndex++] = 0x5e; // up
		if (KbdIndex < sizeof(KbdScan) && (Buttons & GCPAD_DOWN) != 0) KbdScan[KbdIndex++] = 0x5d; // down
		
		if (KbdIndex < sizeof(KbdScan) && CStickY > 64) KbdScan[KbdIndex++] = 0x08; // page up
		if (KbdIndex < sizeof(KbdScan) && CStickY < -64) KbdScan[KbdIndex++] = 0x09; // page down

		if (KbdIndex < sizeof(KbdScan)) {
			if ((Buttons & GCPAD_A) != 0) KbdScan[KbdIndex++] = 0x61; // Enter
			else if ((Buttons & GCPAD_B) != 0) KbdScan[KbdIndex++] = 0x4c; // ESC
			if ((Buttons & GCPAD_X) != 0) KbdScan[KbdIndex++] = 0x47; // F8
			if ((Buttons & GCPAD_Y) != 0) KbdScan[KbdIndex++] = SI_GC_SCANCODE_LOOKUP_START + ('C' - 'A');
			if ((Buttons & GCPAD_Z) != 0) KbdScan[KbdIndex++] = SI_GC_SCANCODE_LOOKUP_START + ('L' - 'A');
		}

		SikbdpGcReceivePart(KbdScan, 0, sizeof(KbdScan));
		return;
	}
	
	// Not in text setup.
	s_MouseReport.X = 0;
	s_MouseReport.Y = 0;
	if (StickX < -64) s_MouseReport.X = -1;
	else if (StickX > 64) s_MouseReport.X = 1;
	if (StickY < -64) s_MouseReport.Y = 1;
	else if (StickY > 64) s_MouseReport.Y = -1;
	s_MouseReport.Buttons = 0;
	if ((Buttons & GCPAD_A) != 0) s_MouseReport.Buttons |= USB_MOUSE_LEFT;
	if ((Buttons & GCPAD_B) != 0) s_MouseReport.Buttons |= USB_MOUSE_RIGHT;
	
	MouReadComplete(&s_MouseReport, 3);
	
	// Additionally allow keyboard.
	if ((Buttons & (GCPAD_L | GCPAD_R)) == (GCPAD_L | GCPAD_R)) {
		// L+R = ctrl+alt+del
		UCHAR ThreeFingerSalute[3] = { 0x56, 0x57, 0x4E };
		SikbdpGcReceivePart(ThreeFingerSalute, 0, sizeof(ThreeFingerSalute));
		UCHAR Key = 0;
		SikbdpGcReceivePart(&Key, 0, sizeof(Key));
	} else {
		// Some form of text entry is *required*, so implement a typical foone quality keyboard:
		// Use c-down and c-up to cycle through scancodes 0x10 (a) to 0x33 (0), and X to accept.
		static UCHAR s_CurrentScan = (0x2A - 0x10); // start at 0x2A (1), to accomodate numeric input
		static BOOLEAN s_Accepted = FALSE;
		static UCHAR s_FrameCount = 0;
		BOOLEAN Changed = FALSE;
		if (CStickY > 64) {
			s_FrameCount++;
			if (s_FrameCount >= CONTROLLER_SCANCODE_FRAME_COUNT) {
				s_FrameCount = 0;
				s_CurrentScan++;
				if (s_CurrentScan == (1 + 0x33 - 0x10)) s_CurrentScan = 0;
				Changed = TRUE;
			}
		} else if (CStickY < -64) {
			s_FrameCount++;
			if (s_FrameCount >= CONTROLLER_SCANCODE_FRAME_COUNT) {
				s_FrameCount = 0;
				if (s_CurrentScan == 0) s_CurrentScan = 0x33 - 0x10;
				else s_CurrentScan--;
				Changed = TRUE;
			}
		}
		
		if ((Buttons & GCPAD_X) != 0) s_Accepted = TRUE;
		
		if (Changed) {
			if (s_Accepted) {
				s_CurrentScan = (0x2A - 0x10);
				s_Accepted = FALSE;
				UCHAR Key = s_CurrentScan + 0x10;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
				Key = 0;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
			} else {
				// send backspace first
				UCHAR Key = 0x50;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
				Key = 0;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
				Key = s_CurrentScan + 0x10;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
				Key = 0;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
			}
		}
	}
}

static void SikbdpPollCallback64Kbd(ULONG Channel, PUCHAR Data) {
	// Randnet keyboard.
	UCHAR flags = Data[6];

	// Three shorts, indicating up to three keys being pressed at same time.
	USHORT ScanCodes[3] = {
		((USHORT)Data[0] << 8) | Data[1],
		((USHORT)Data[2] << 8) | Data[3],
		((USHORT)Data[4] << 8) | Data[5]
	};
	BOOLEAN oldCaps = s_CapsLock;
	BOOLEAN oldNum = s_NumLock;
	Sikbdp64ReceivePart(ScanCodes, 0, 3, flags);
	if (s_CapsLock != oldCaps || s_NumLock != oldNum) {
		ULONG command = SI_CMD_READ_64KBD << 8;
		command |= KBD64_LED_POWER;
		if (s_NumLock) command |= KBD64_LED_NUM;
		if (s_CapsLock) command |= KBD64_LED_CAPS;
		SiTransferPoll(BIT(Channel), command, 2);
	}
}

static void SikbdpPollCallback64Mouse(ULONG Channel, PUCHAR Data) {
	// N64 mouse as bundled with Mario Artist Paint Studio.
	// 16 bit buttons ; 16 bit analog stick (8 bit x/y)
	USHORT Buttons = ((USHORT)Data[0] << 8) | Data[1];
	signed char StickY = (char)((UCHAR)(Data[3]));
	signed char StickX = (char)((UCHAR)(Data[2]));
	
	s_MouseReport.X = StickX;
	s_MouseReport.Y = StickY;
	s_MouseReport.Buttons = 0;
	if ((Buttons & N64MOUSE_LEFT) != 0) s_MouseReport.Buttons |= USB_MOUSE_LEFT;
	if ((Buttons & N64MOUSE_RIGHT) != 0) s_MouseReport.Buttons |= USB_MOUSE_RIGHT;
	
	MouReadComplete(&s_MouseReport, 3);
}

static void SikbdpPollCallback64Con(ULONG Channel, PUCHAR Data) {
	// N64 controller.
	// 16 bit buttons ; 16 bit analog stick (8 bit x/y)
	USHORT Buttons = ((USHORT)Data[0] << 8) | Data[1];
	signed char StickY = (char)((UCHAR)(Data[3]));
	signed char StickX = (char)((UCHAR)(Data[2]));
	
	if (s_InTextSetup) {
		UCHAR KbdScan[10] = { 0 };
		UCHAR KbdIndex = 0;

		if (KbdIndex < sizeof(KbdScan) && StickX > 64) KbdScan[KbdIndex++] = 0x5f; // right
		if (KbdIndex < sizeof(KbdScan) && StickX < -64) KbdScan[KbdIndex++] = 0x5c; // left
		if (KbdIndex < sizeof(KbdScan) && StickY > 64) KbdScan[KbdIndex++] = 0x5e; // up
		if (KbdIndex < sizeof(KbdScan) && StickY < -64) KbdScan[KbdIndex++] = 0x5d; // down

		if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_RIGHT) != 0) KbdScan[KbdIndex++] = 0x5f; // right
		if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_LEFT) != 0) KbdScan[KbdIndex++] = 0x5c; // left
		if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_UP) != 0) KbdScan[KbdIndex++] = 0x5e; // up
		if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_DOWN) != 0) KbdScan[KbdIndex++] = 0x5d; // down
		
		if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_CUP) != 0) KbdScan[KbdIndex++] = 0x08; // page up
		if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_CDOWN) != 0) KbdScan[KbdIndex++] = 0x09; // page down

		if (KbdIndex < sizeof(KbdScan)) {
			if ((Buttons & N64PAD_A) != 0) KbdScan[KbdIndex++] = 0x61; // Enter
			else if ((Buttons & N64PAD_B) != 0) KbdScan[KbdIndex++] = 0x4c; // ESC
			if ((Buttons & N64PAD_Z) != 0) KbdScan[KbdIndex++] = 0x47; // F8
			if ((Buttons & N64PAD_L) != 0) KbdScan[KbdIndex++] = SI_GC_SCANCODE_LOOKUP_START + ('C' - 'A');
			if ((Buttons & N64PAD_R) != 0) KbdScan[KbdIndex++] = SI_GC_SCANCODE_LOOKUP_START + ('L' - 'A');
		}

		SikbdpGcReceivePart(KbdScan, 0, sizeof(KbdScan));
		return;
	}
	
	s_MouseReport.X = 0;
	s_MouseReport.Y = 0;
	if (StickX < -64) s_MouseReport.X = -1;
	else if (StickX > 64) s_MouseReport.X = 1;
	if (StickY < -64) s_MouseReport.Y = 1;
	else if (StickY > 64) s_MouseReport.Y = -1;
	s_MouseReport.Buttons = 0;
	if ((Buttons & N64PAD_A) != 0) s_MouseReport.Buttons |= USB_MOUSE_LEFT;
	if ((Buttons & N64PAD_B) != 0) s_MouseReport.Buttons |= USB_MOUSE_RIGHT;
	
	MouReadComplete(&s_MouseReport, 3);
	
	// Additionally allow keyboard.
	if ((Buttons & (N64PAD_L | N64PAD_R)) == (N64PAD_L | N64PAD_R)) {
		// L+R = ctrl+alt+del
		UCHAR ThreeFingerSalute[3] = { 0x56, 0x57, 0x4E };
		SikbdpGcReceivePart(ThreeFingerSalute, 0, sizeof(ThreeFingerSalute));
		UCHAR Key = 0;
		SikbdpGcReceivePart(&Key, 0, sizeof(Key));
	} else {
		// Some form of text entry is *required*, so implement a typical foone quality keyboard:
		// Use c-down and c-up to cycle through scancodes 0x10 (a) to 0x33 (0), and START to accept.
		static UCHAR s_CurrentScan = (0x2A - 0x10); // start at 0x2A (1), to accomodate numeric input
		static BOOLEAN s_Accepted = FALSE;
		static UCHAR s_FrameCount = 0;
		BOOLEAN Changed = FALSE;
		if ((Buttons & N64PAD_CUP) != 0) {
			s_FrameCount++;
			if (s_FrameCount >= CONTROLLER_SCANCODE_FRAME_COUNT) {
				s_FrameCount = 0;
				s_CurrentScan++;
				if (s_CurrentScan == (1 + 0x33 - 0x10)) s_CurrentScan = 0;
				Changed = TRUE;
			}
		} else if ((Buttons & N64PAD_CDOWN) != 0) {
			s_FrameCount++;
			if (s_FrameCount >= CONTROLLER_SCANCODE_FRAME_COUNT) {
				s_FrameCount = 0;
				if (s_CurrentScan == 0) s_CurrentScan = 0x33 - 0x10;
				else s_CurrentScan--;
				Changed = TRUE;
			}
		}
		
		if ((Buttons & N64PAD_START) != 0) s_Accepted = TRUE;
		
		if (Changed) {
			if (s_Accepted) {
				s_CurrentScan = (0x2A - 0x10);
				s_Accepted = FALSE;
				UCHAR Key = s_CurrentScan + 0x10;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
				Key = 0;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
			} else {
				// send backspace first
				UCHAR Key = 0x50;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
				Key = 0;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
				Key = s_CurrentScan + 0x10;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
				Key = 0;
				SikbdpGcReceivePart(&Key, 0, sizeof(Key));
			}
		}
	}
}

static BOOLEAN SikbdpInitPollingGc(ULONG channel, USHORT type) {
	type &= 0x1ff;

	if ((type & 0x1f0) == SI_GC_TYPE_KEYBOARD || (type & 0x1f0) == SI_GC_TYPE_KEYBOARD2) {
		// Keyboard controller.
		if (!NT_SUCCESS(SiPollSetCallback(channel, SikbdpPollCallbackGcKbd))) return FALSE;
		if (!NT_SUCCESS(SiTransferPoll(BIT(channel), SI_CMD_READ_GCKBD, 1))) return FALSE;
		return TRUE;
	}
	
	// assume everything else is some kind of controller.
	ULONG command = (SI_CMD_READ_GC << 16) | (0x03 << 8);
	if (!NT_SUCCESS(SiPollSetCallback(channel, SikbdpPollCallbackGcCon))) return FALSE;
	if (!NT_SUCCESS(SiTransferPoll(BIT(channel), command, 3))) return FALSE;
	return TRUE;
}

static BOOLEAN SikbdpInitPolling64(ULONG channel, USHORT type) {
	if (type == SI_N64_TYPE_KEYBOARD) {
		// Randnet keyboard controller.
		ULONG command = SI_CMD_READ_64KBD << 8;
		command |= KBD64_LED_POWER;
		if (s_CapsLock) command |= KBD64_LED_CAPS;
		if (!NT_SUCCESS(SiPollSetCallback(channel, SikbdpPollCallback64Kbd))) return FALSE;
		if (!NT_SUCCESS(SiTransferPoll(BIT(channel), command, 2))) return FALSE;
		return TRUE;
	}
	
	if (type == SI_N64_TYPE_MOUSE) {
		// N64 mouse as bundled with Mario Artist Paint Studio.
		if (!NT_SUCCESS(SiPollSetCallback(channel, SikbdpPollCallback64Mouse))) return FALSE;
		if (!NT_SUCCESS(SiTransferPoll(BIT(channel), SI_CMD_READ_64, 1))) return FALSE;
		return TRUE;
	}
	
	if (type == SI_N64_TYPE_CONTROLLER) {
		// normal N64 controller.
		if (!NT_SUCCESS(SiPollSetCallback(channel, SikbdpPollCallback64Con))) return FALSE;
		if (!NT_SUCCESS(SiTransferPoll(BIT(channel), SI_CMD_READ_64, 1))) return FALSE;
		return TRUE;
	}
	
	return FALSE;
}

static BOOLEAN SikbdpInitPolling(ULONG channel) {
	if (!SiDeviceTypeValid(s_ConnectedDevices[channel])) return FALSE;

	USHORT ControllerType = (s_ConnectedDevices[channel].Identifier >> 11) & 3;
	switch (ControllerType) {
	case SI_CONTROLLER_TYPE_N64:
		// N64 controller.
		return SikbdpInitPolling64(channel, s_ConnectedDevices[channel].Identifier);
		break;
	case SI_CONTROLLER_TYPE_GC:
		// GC controller.
		return SikbdpInitPollingGc(channel, s_ConnectedDevices[channel].Identifier);
		break;
	}
	
	return FALSE;
}