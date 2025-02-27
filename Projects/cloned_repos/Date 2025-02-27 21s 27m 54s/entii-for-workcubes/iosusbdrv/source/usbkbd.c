// IOS USB KBD driver.
#define DEVL 1
#include <ntddk.h>
#include "iosusbkbd.h"
#include "iosapi.h"
#include "asynctimer.h"

static IOS_HANDLE s_hUsbKbd = IOS_HANDLE_INVALID;
//static HANDLE s_KbdReadThread = NULL;
static IOSUSBKBDEvent s_Event = {0};
static IOSUSBKBDLed s_Led = {0};
static ULONG s_KbdId = 0;
static KTIMER s_KbdTimer;
static KDPC s_KbdDpc;

static const char sc_UsbKbd[] ARC_ALIGNED(32) = STRING_BYTESWAP("/dev/usb/kbd");

/*
static void IukpReadThread(PVOID Context) {
	(void)Context;
	
	while (s_hUsbKbd != IOS_HANDLE_INVALID) {
		// In a thread for this.
		// Only care about one ioctl.
		// So use blocking sync ioctl :)
		NTSTATUS Status = HalIopIoctl(s_hUsbKbd, USBKBD_IOCTL_READEVENT, NULL, 0, &s_Event, sizeof(s_Event));
		if (NT_SUCCESS(Status)) {
			if (s_Event.type == KEYBOARD_ATTACH) {
				if (s_KbdId == 0) s_KbdId = s_Event.id;
			} else if (s_Event.type == KEYBOARD_DETACH) {
				if (s_KbdId == s_Event.id) s_KbdId = 0;
			} else if (s_Event.type == KEYBOARD_EVENT) {
				KbdReadComplete(&s_Event.report);
			}
		}
	}
	PsTerminateSystemThread(STATUS_SUCCESS);
}
*/

static void IukpReadCallback(NTSTATUS Status, ULONG Result, PVOID Context);

static void IukpTimerCallback(PKDPC Dpc, PVOID DeferredContext, PVOID SystemArgument1, PVOID SystemArgument2) {
	NTSTATUS Status = HalIopIoctlAsyncDpc(
		s_hUsbKbd,
		USBKBD_IOCTL_READEVENT,
		NULL,
		0,
		&s_Event,
		sizeof(s_Event),
		IOCTL_SWAP_NONE, IOCTL_SWAP_OUTPUT,
		IukpReadCallback,
		NULL
	);
	
	if (!NT_SUCCESS(Status)) {
		AsyncTimerSet(&s_KbdTimer, &s_KbdDpc);
	}
}

static void IukpReadCallback(NTSTATUS Status, ULONG Result, PVOID Context) {
	if (NT_SUCCESS(Status)) {
		if (s_Event.type == KEYBOARD_ATTACH) {
			if (s_KbdId == 0) s_KbdId = s_Event.id;
		} else if (s_Event.type == KEYBOARD_DETACH) {
			if (s_KbdId == s_Event.id) s_KbdId = 0;
		} else if (s_Event.type == KEYBOARD_EVENT) {
			KbdReadComplete(&s_Event.report);
		}
	}
	IukpTimerCallback(NULL, NULL, NULL, NULL);
}

static void IukpSetLedsCallback(NTSTATUS Status, ULONG Result, PVOID Context) {
	// no operation
}

NTSTATUS IukSetLeds(UCHAR UsbIndicators) {
	if (s_hUsbKbd == IOS_HANDLE_INVALID) return STATUS_NO_SUCH_DEVICE;
	if (s_KbdId == 0) return STATUS_SUCCESS;
	
	s_Led.id = s_KbdId;
	s_Led.LedState = UsbIndicators;
	
	NTSTATUS Status = HalIopWriteAsyncDpc(
		s_hUsbKbd,
		&s_Led,
		KBD_LED_SIZEOF,
		IOCTL_SWAP_NONE,
		IukpSetLedsCallback,
		NULL
	);
	return Status;
}

NTSTATUS IukInit(void) {
	if (s_hUsbKbd != IOS_HANDLE_INVALID) return STATUS_SUCCESS;
	// Open IOS keyboard device.
	NTSTATUS Status = HalIopOpen(sc_UsbKbd, IOSOPEN_NONE, &s_hUsbKbd);
	if (!NT_SUCCESS(Status)) return Status;
	
	// Initialise the timer and DPC.
	KeInitializeDpc(&s_KbdDpc, IukpTimerCallback, NULL);
	KeInitializeTimer(&s_KbdTimer);
	// Start the async DPC requests.
	Status = HalIopIoctlAsyncDpc(
		s_hUsbKbd,
		USBKBD_IOCTL_READEVENT,
		NULL,
		0,
		&s_Event,
		sizeof(s_Event),
		IOCTL_SWAP_NONE, IOCTL_SWAP_OUTPUT,
		IukpReadCallback,
		NULL
	);
	
#if 0
	Status = PsCreateSystemThread(
		&s_KbdReadThread,
		0,
		NULL,
		NULL,
		NULL,
		IukpReadThread,
		NULL
	);
#endif
	
	if (!NT_SUCCESS(Status)) {
		HalIopClose(s_hUsbKbd);
		s_hUsbKbd = IOS_HANDLE_INVALID;
	}
	return Status;
}