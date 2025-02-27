#pragma once
#include "runtime.h"

typedef struct ARC_BE {
	UCHAR Modifiers;
	UCHAR Reserved;
	UCHAR KeyCode[6];
} USB_KBD_REPORT, *PUSB_KBD_REPORT;

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

// Keyboard device object.
extern PDEVICE_OBJECT KbdDeviceObject;

// Set keyboard LEDs (IOS /dev/usb/kbd)
NTSTATUS IukSetLeds(UCHAR Indicators);
// Set keyboard LEDs (IOS USBv5 low-level)
NTSTATUS UlkSetLeds(UCHAR Indicators);

// Called from lower-level driver when keyboard input data read complete.
void KbdReadComplete(PUSB_KBD_REPORT Report);

// USB keyboard ioctl implementation...
NTSTATUS KbdIoctl(PDEVICE_OBJECT Device, PIRP Irp);

// Initialise USB keyboard driver.
NTSTATUS KbdInit(PDRIVER_OBJECT Driver, PUNICODE_STRING RegistryPath);