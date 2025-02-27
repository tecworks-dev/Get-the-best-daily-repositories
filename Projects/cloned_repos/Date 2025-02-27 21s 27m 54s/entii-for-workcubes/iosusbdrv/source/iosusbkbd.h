#pragma once
#include "keyboard.h"

enum {
	USBKBD_IOCTL_READEVENT = 0
};

typedef enum {
	KEYBOARD_ATTACH,
	KEYBOARD_DETACH,
	KEYBOARD_EVENT
} IOSUSBKBDType;

typedef struct ARC_BE {
	ULONG type;
	ULONG id;
	USB_KBD_REPORT report;
} IOSUSBKBDEvent;

typedef struct {
	UCHAR Padding[3];
	UCHAR LedState;
	ULONG id;
} IOSUSBKBDLed;

#define KBD_LED_SIZEOF (sizeof(IOSUSBKBDLed) - __builtin_offsetof(IOSUSBKBDLed, LedState))

NTSTATUS IukInit(void);