// SI keyboard driver.
// Supports GC controller, GC keyboard, N64 controller, N64 keyboard

#include <stdio.h>
#include <memory.h>
#include "arc.h"
#include "types.h"
#include "runtime.h"
#include "si.h"
#include "si_kbd.h"
#include "kbd_high.h"

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
	SI_N64_TYPE_MOUSE = 0x200, // not used in ARC firmware.
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
};

enum {
	KBD64_LED_NUM = ARC_BIT(0),
	KBD64_LED_CAPS = ARC_BIT(1),
	KBD64_LED_POWER = ARC_BIT(2)
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
static bool s_CapsLock = false;

static bool SikbdpUsbAddKey(PUSB_KBD_REPORT Report, UCHAR UsbScanCode) {
	UCHAR FreeSlot = sizeof(Report->KeyCode);
	for (UCHAR i = 0; i < sizeof(Report->KeyCode); i++) {
		if (Report->KeyCode[i] == UsbScanCode) return true;
		if (Report->KeyCode[i] == 0 && FreeSlot >= sizeof(Report->KeyCode))
			FreeSlot = i;
	}

	if (FreeSlot >= sizeof(Report->KeyCode)) return false;
	Report->KeyCode[FreeSlot] = UsbScanCode;
	return true;
}

static bool SikbdpUsbRemoveKey(PUSB_KBD_REPORT Report, UCHAR UsbScanCode) {
	bool Changed = false;
	for (UCHAR i = 0; i < sizeof(Report->KeyCode); i++) {
		if (Report->KeyCode[i] != UsbScanCode) continue;
		Report->KeyCode[i] = 0;
		Changed = true;
	}
	return Changed;
}

static bool SikbdpUpdateKey(PUSB_KBD_REPORT Report, UCHAR UsbScanCode, bool Released) {
	if (UsbScanCode <= 3) return false;
	if (Released) return SikbdpUsbRemoveKey(Report, UsbScanCode);
	return SikbdpUsbAddKey(Report, UsbScanCode);
}

static bool SikbdpUsbUpdateModifier(PUSB_KBD_REPORT Report, UCHAR UsbModifierBit, bool Released) {
	// check if the bit is already correct
	bool IsSet = (Report->Modifiers & UsbModifierBit) != 0;
	if (IsSet == !Released) return false;

	// bit is incorrect, so flip it
	Report->Modifiers ^= UsbModifierBit;
	return true;
}

static void SikbdpReset(ULONG channel) {
	// Send reset command to device, use the response value as the actual device type.
	JOYBUS_DEVICE_TYPE DeviceType;
	DeviceType.Value = 0;
	for (ULONG attempt = 0; attempt < 10; attempt++) {
		udelay(1);
		if (!SiTransferByteSync(channel, SI_CMD_RESET, &DeviceType, 3)) {
			s_ConnectedDevices[channel].Value = 0xFFFFFFFF;
			continue;
		}
		s_ConnectedDevices[channel] = DeviceType;
		return;
	}
}

void SikbdInit(void) {
	// Get device type for each connected device.
	for (ULONG channel = 0; channel < SI_CHANNEL_COUNT; channel++) {
		for (ULONG attempt = 0; attempt < 10; attempt++) {
			s_ConnectedDevices[channel] = SiGetDeviceTypeReset(channel);
			if (SiDeviceTypeValid(s_ConnectedDevices[channel])) break;
			udelay(1);
		}
	}

	for (ULONG channel = 0; channel < SI_CHANNEL_COUNT; channel++) {
		if (!SiDeviceTypeValid(s_ConnectedDevices[channel])) continue;

		SikbdpReset(channel);
		//USHORT ControllerType = (s_ConnectedDevices[channel].Identifier >> 11) & 3;
		//printf("[KBD%d] - type %x (%04x)\r\n", channel, ControllerType, s_ConnectedDevices[channel].Identifier);
	}

	// All done.
}

static bool Sikbdp64UpdateModifiers(PUSB_KBD_REPORT Report, UCHAR ScanCode, bool Released) {
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
		return false;
	}
}

static bool Sikbdp64KeyIsModifier(UCHAR UsbScanCode) {
	switch (UsbScanCode) {
	case KEY_LEFTCTRL:
	case KEY_LEFTSHIFT:
	case KEY_LEFTALT:
	case KEY_LEFTMETA:
	case KEY_RIGHTSHIFT:
	case KEY_RIGHTMETA:
		return true;
	default:
		return false;
	}
	return false;
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
static bool SikbdpConvert64Usb(PUSB_KBD_REPORT Report, PUSHORT Data, UCHAR Offset, UCHAR Length, UCHAR Flags) {
	if ((Flags & ARC_BIT(4))) return false; // Error flag set, leave keyboard data untouched.

	bool Changed = false;

	// Home key is bit 0 of the flags instead of its own scancode for some reason.
	if (Flags & ARC_BIT(0)) {
		Changed = SikbdpUpdateKey(Report, KEY_HOME, false);
	}

	for (UCHAR i = Offset; i < Length; i++) {
		UCHAR ScanCode = Sikbdp64ConvertScancode(Data[i]);
		if (ScanCode <= 3) continue;
		if (ScanCode == KEY_CAPSLOCK) s_CapsLock ^= 1;
		bool ChangedThis = false;
		if (Sikbdp64KeyIsModifier(ScanCode)) {
			ChangedThis = Sikbdp64UpdateModifiers(Report, ScanCode, false);
		}
		else {
			// Not a modifier
			ChangedThis = SikbdpUpdateKey(Report, ScanCode, false);
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

	if (Changed) {
		KBDOnEvent(&s_Report);
	}
	else if (s_Report.KeyCode[0] == 0) KbdHighZeroLast();
}

static bool SikbdpGcKeyIsModifier(UCHAR GcScanCode) {
	switch (GcScanCode) {
	case GC_KEY_LCTRL:
	case GC_KEY_LSHIFT:
	case GC_KEY_LALT:
	case GC_KEY_LWIN:
	case GC_KEY_RSHIFT:
	case GC_KEY_RWIN:
		return true;
	default:
		return false;
	}
	return false;
}

static bool SikbdpGcUpdateModifiers(PUSB_KBD_REPORT Report, UCHAR GcScanCode, bool Released) {
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
		return false;
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

static bool SikbdpConvertGcUsb(PUSB_KBD_REPORT Report, PUCHAR Data, UCHAR Offset, UCHAR Length) {
	bool Changed = false;
	for (UCHAR i = Offset; i < Length; i++) {
		if (Data[i] < 3 || Data[i] >= SI_GC_SCANCODE_END) continue;
		UCHAR ScanCode = Data[i];
		bool ChangedThis = false;
		if (SikbdpGcKeyIsModifier(ScanCode)) {
			ChangedThis = SikbdpGcUpdateModifiers(Report, ScanCode, false);
		}
		else {
			// Not a modifier
			ChangedThis = SikbdpUpdateKey(Report, SikbdpGcConvertScancode(ScanCode), false);
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

	if (Changed) {
		KBDOnEvent(&s_Report);
	}
	else if (s_Report.KeyCode[0] == 0) KbdHighZeroLast();
}

static void SikbdpPollChannelGc(ULONG channel, USHORT type) {
	type &= 0x1ff;

	ULONG Buffer[2];

	if ((type & 0x1f0) == SI_GC_TYPE_KEYBOARD || (type & 0x1f0) == SI_GC_TYPE_KEYBOARD2) {
		// Keyboard controller.
		if (!SiTransferByteSync(channel, SI_CMD_READ_GCKBD, Buffer, sizeof(Buffer))) {
			// Transfer failed, reset the controller...
			SikbdpReset(channel);
			return;
		}

		// Pull out scancodes...
		UCHAR ScanCodes[3] = { Buffer[1] >> 24, Buffer[1] >> 16, Buffer[1] >> 8 };
		// Three bytes, indicating up to three keys being pressed at same time.
		SikbdpGcReceivePart(ScanCodes, 0, 3);
		return;
	}

	// Assume everything else is some kind of controller.
	// Dolphin doesn't emulate this properly, but it starts up in mode 3 anyway, so as long as we use that it's fine;
	// Three bytes - cmd, mode, rumble;
	UCHAR cmdBuf[] = { SI_CMD_READ_GC, 0x03, 0x00 };
	// 64 bit response
	if (!SiTransferSync(channel, cmdBuf, sizeof(cmdBuf), Buffer, sizeof(Buffer))) {
		// Transfer failed, reset the controller...
		SikbdpReset(channel);
		return;
	}

	// 16 bit buttons ; 16 bit analog stick (8 bit x/y) ; 32 bit analog triggers (8 bit x/y/l/r)
	USHORT Buttons = (Buffer[0] >> 16);

	signed char StickY = (char)((UCHAR)(Buffer[0] >> 0) - 0x80);
	signed char StickX = (char)((UCHAR)(Buffer[0] >> 8) - 0x80);
	signed char CStickY = (char)((UCHAR)(Buffer[1] >> 16) - 0x80);
	signed char CStickX = (char)((UCHAR)(Buffer[1] >> 24) - 0x80);

	// Convert to keyboard scancodes.
	// Up to two of analog stick, two dpad plus one of A/B + X.
	// (For the NT text setup controller-as-keyboard driver we'll also need to support pressing page down, probably trigger + direction can be used for that.)
	UCHAR KbdScan[6] = { 0 };
	UCHAR KbdIndex = 0;

	if (KbdIndex < sizeof(KbdScan) && StickX > 64) KbdScan[KbdIndex++] = 0x5f; // right
	if (KbdIndex < sizeof(KbdScan) && StickX < -64) KbdScan[KbdIndex++] = 0x5c; // left
	if (KbdIndex < sizeof(KbdScan) && StickY > 64) KbdScan[KbdIndex++] = 0x5e; // up
	if (KbdIndex < sizeof(KbdScan) && StickY < -64) KbdScan[KbdIndex++] = 0x5d; // down

	if (KbdIndex < sizeof(KbdScan) && (Buttons & GCPAD_RIGHT) != 0) KbdScan[KbdIndex++] = 0x5f; // right
	if (KbdIndex < sizeof(KbdScan) && (Buttons & GCPAD_LEFT) != 0) KbdScan[KbdIndex++] = 0x5c; // left
	if (KbdIndex < sizeof(KbdScan) && (Buttons & GCPAD_UP) != 0) KbdScan[KbdIndex++] = 0x5e; // up
	if (KbdIndex < sizeof(KbdScan) && (Buttons & GCPAD_DOWN) != 0) KbdScan[KbdIndex++] = 0x5d; // down

	if (KbdIndex < sizeof(KbdScan)) {
		if ((Buttons & GCPAD_A) != 0) KbdScan[KbdIndex++] = 0x61; // Enter
		else if ((Buttons & GCPAD_B) != 0) KbdScan[KbdIndex++] = 0x4c; // ESC
		if ((Buttons & GCPAD_X) != 0) KbdScan[KbdIndex++] = SI_GC_SCANCODE_LOOKUP_START + ('S' - 'A');
	}

	SikbdpGcReceivePart(KbdScan, 0, sizeof(KbdScan));
}

// N64 main poll function after GC so we can convert to simpler gamecube scancodes for N64 controller...
static void SikbdpPollChannel64(ULONG channel, USHORT type) {
	if (type == SI_N64_TYPE_KEYBOARD) {
		// Randnet keyboard.
		UCHAR cmdBuf[] = { SI_CMD_READ_64KBD, KBD64_LED_POWER };
		if (s_CapsLock) cmdBuf[1] |= KBD64_LED_CAPS;
		// 56 bit response - 3 * 16 bit scancodes + 8-bit bitflags
		ULONG KbdBuf[2];
		if (!SiTransferSync(channel, cmdBuf, sizeof(cmdBuf), KbdBuf, sizeof(KbdBuf) - 1)) {
			// Transfer failed, reset the controller...
			SikbdpReset(channel);
			return;
		}

		UCHAR flags = KbdBuf[1] >> 8;

		// Three shorts, indicating up to three keys being pressed at same time.
		USHORT ScanCodes[3] = { KbdBuf[0] >> 16, KbdBuf[0], KbdBuf[1] >> 16 };
		Sikbdp64ReceivePart(ScanCodes, 0, 3, flags);
		return;
	}

	if (type != SI_N64_TYPE_CONTROLLER) {
		return;
	}

	// 32 bit response
	ULONG Buffer;
	if (!SiTransferByteSync(channel, SI_CMD_READ_64, &Buffer, sizeof(Buffer))) {
		// Transfer failed, reset the controller...
		SikbdpReset(channel);
		return;
	}

	// 16 bit buttons ; 16 bit analog stick (8 bit x/y)
	USHORT Buttons = Buffer >> 16;
	signed char StickY = (char)((UCHAR)(Buffer >> 0));
	signed char StickX = (char)((UCHAR)(Buffer >> 8));

	// No need to convert the analog stick data alike GC, N64 controllers say signed char, 0==neutral.

	// Convert to GC keyboard scancodes.
	// Up to two of analog stick, two dpad plus one of A/B + Z.
	// (For the NT text setup controller-as-keyboard driver we'll also need to support pressing page down, on N64 controllers c-up or c-down can be used for that...)
	UCHAR KbdScan[6] = { 0 };
	UCHAR KbdIndex = 0;

	if (KbdIndex < sizeof(KbdScan) && StickX > 64) KbdScan[KbdIndex++] = 0x5f; // right
	if (KbdIndex < sizeof(KbdScan) && StickX < -64) KbdScan[KbdIndex++] = 0x5c; // left
	if (KbdIndex < sizeof(KbdScan) && StickY > 64) KbdScan[KbdIndex++] = 0x5e; // up
	if (KbdIndex < sizeof(KbdScan) && StickY < -64) KbdScan[KbdIndex++] = 0x5d; // down

	if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_RIGHT) != 0) KbdScan[KbdIndex++] = 0x5f; // right
	if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_LEFT) != 0) KbdScan[KbdIndex++] = 0x5c; // left
	if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_UP) != 0) KbdScan[KbdIndex++] = 0x5e; // up
	if (KbdIndex < sizeof(KbdScan) && (Buttons & N64PAD_DOWN) != 0) KbdScan[KbdIndex++] = 0x5d; // down

	if (KbdIndex < sizeof(KbdScan)) {
		if ((Buttons & N64PAD_A) != 0) KbdScan[KbdIndex++] = 0x61; // Enter
		else if ((Buttons & N64PAD_B) != 0) KbdScan[KbdIndex++] = 0x4c; // ESC
		if ((Buttons & N64PAD_Z) != 0) KbdScan[KbdIndex++] = SI_GC_SCANCODE_LOOKUP_START + ('S' - 'A');
	}

	SikbdpGcReceivePart(KbdScan, 0, sizeof(KbdScan));
}

static void SikbdpPollChannel(ULONG channel) {
	if (!SiDeviceTypeValid(s_ConnectedDevices[channel])) return;

	USHORT ControllerType = (s_ConnectedDevices[channel].Identifier >> 11) & 3;
	switch (ControllerType) {
	case SI_CONTROLLER_TYPE_N64:
		// N64 controller.
		SikbdpPollChannel64(channel, s_ConnectedDevices[channel].Identifier);
		break;
	case SI_CONTROLLER_TYPE_GC:
		// GC controller.
		SikbdpPollChannelGc(channel, s_ConnectedDevices[channel].Identifier);

		break;
	}
}

void SikbdPoll(void) {
	for (ULONG channel = 0; channel < SI_CHANNEL_COUNT; channel++)
		SikbdpPollChannel(channel);
}