// RVL/Cafe USB HID by USBv5
#include <stdio.h>
#include <memory.h>
#include "arc.h"
#include "types.h"
#include "runtime.h"
#include "pxi.h"
#include "timer.h"
#include "ios_usb.h"
#include "kbd_high.h"

static IOS_USB_HANDLE s_UsbHandle;
static PUSB_KBD_REPORT s_ReportRead = NULL;
static PU8_AS_32 s_ReportWrite = NULL;
static ULONG s_InputReportAsync;
static UCHAR s_Interface;
static UCHAR s_Endpoint;
static bool s_InputReportAsyncInProgress = false;

static void ZeroMemory32(void* buffer, ULONG length) {
	if ((length & 3) != 0) {
		memset(buffer, 0, length);
		return;
	}
	length /= sizeof(ULONG);
	PULONG buf32 = (PULONG)buffer;
	for (ULONG i = 0; i < length; i++) buf32[i] = 0;
}

static LONG UlkSetLeds(UCHAR UsbIndicators) {
	if (s_ReportWrite == NULL) return -1;

	U8_AS_32 Usb32 = { .Long = 0 };
	Usb32.Char = UsbIndicators;
	s_ReportWrite->Long = Usb32.Long;
	return UlTransferControlMessage(
		s_UsbHandle,
		USB_REQTYPE_INTERFACE_SET,
		USB_REQ_SETREPORT,
		USB_REPTYPE_OUTPUT << 8,
		s_Interface,
		sizeof(UsbIndicators),
		s_ReportWrite
	);
}

static LONG UlkpStartReceiveInputReport(void) {
	s_InputReportAsyncInProgress = false;
	//UlkSetLeds(0);
	LONG Status = UlTransferInterruptMessageAsync(
		s_UsbHandle,
		s_Endpoint,
		sizeof(*s_ReportRead),
		s_ReportRead
	);
	if (Status < 0) {
		return Status;
	}
	s_InputReportAsync = Status;
	s_InputReportAsyncInProgress = true;
	return Status;
}

static LONG UlkpGetProtocol(PUCHAR Protocol) {
	PUCHAR Buffer = PxiIopAlloc(sizeof(*Protocol));
	if (Buffer == NULL) return -1;
	LONG Status = UlTransferControlMessage(
		s_UsbHandle,
		USB_REQTYPE_INTERFACE_GET,
		USB_REQ_GETPROTOCOL,
		0,
		s_Interface,
		1,
		Buffer
	);
	if (Status >= 0) *Protocol = *Buffer;
	PxiIopFree(Buffer);
	return Status;
}

static LONG UlkpSetProtocol(UCHAR Protocol) {
	return UlTransferControlMessage(
		s_UsbHandle,
		USB_REQTYPE_INTERFACE_SET,
		USB_REQ_SETPROTOCOL,
		Protocol,
		s_Interface,
		0,
		NULL
	);
}

void UlkPoll(void) {
	if (s_ReportRead == NULL) return;

	if (s_InputReportAsyncInProgress && PxiIopIoctlvAsyncActive(s_InputReportAsync)) {
		// Try to check if the async ipc has finished.
		LONG Result;
		PVOID Context;	
		if (!PxiIopIoctlvAsyncPoll(s_InputReportAsync, &Result, &Context)) {
			// Not completed yet.
			return;
		}
		// Completed, free the usb-api context
		Context = UlGetPassedAsyncContext(Context);

		if (Result >= 0) {
			// IPC succeeded, we now have a report to pass on
			KBDOnEvent(s_ReportRead);
		}
	}

	// Async not in progress, so start off another one
	UlkpStartReceiveInputReport();
}

void UlkShutdown(void) {
	if (s_ReportRead != NULL) return;
	UlClearHalt(s_UsbHandle);
	if (s_InputReportAsyncInProgress && PxiIopIoctlvAsyncActive(s_InputReportAsync)) {
		// Try to check if the async ipc has finished.
		LONG Result;
		PVOID Context;
		while (!PxiIopIoctlvAsyncPoll(s_InputReportAsync, &Result, &Context)) udelay(100);
	}
	UlCloseDevice(s_UsbHandle);
	PxiIopFree(s_ReportRead);
	s_ReportRead = NULL;
}

LONG UlkInit(void) {
	if (s_ReportRead != NULL) return 0;

	PIOS_USB_DEVICE_ENTRY Entries = (PIOS_USB_DEVICE_ENTRY)
		PxiIopAlloc(sizeof(IOS_USB_DEVICE_ENTRY_MAX));
	if (Entries == NULL) return -1;

	ZeroMemory32(Entries, sizeof(IOS_USB_DEVICE_ENTRY_MAX));

	UCHAR NumHid;
	UlGetDeviceList(Entries, USB_COUNT_DEVICES, USB_CLASS_HID, &NumHid);

	bool Found = false;
	UCHAR Configuration, AltInterface;
	USHORT MaxPacketSize;

	for (ULONG i = 0; !Found && i < NumHid; i++) {
		if (Entries[i].VendorId == 0 || Entries[i].ProductId == 0) {
			continue;
		}

		//printf("%d: %08x:%08x\r\n", i, Entries[i].VendorId, Entries[i].ProductId);

		IOS_USB_HANDLE DeviceHandle = Entries[i].DeviceHandle;
		LONG Status = UlOpenDevice(DeviceHandle);
		if (Status < 0) {
			//printf("%d: UlOpenDevice failed %d\r\n", i, Status);
			continue;
		}
		do {
			USB_DEVICE_DESC Descriptors;
			Status = UlGetDescriptors(DeviceHandle, &Descriptors);
			if (Status < 0) {
				//printf("%d: UlGetDescriptors failed %d\r\n", i, Status);
				continue;
			}

			//printf("%d: bNumConfigurations %d\r\n", i, Descriptors.Device.bNumConfigurations);
			if (Descriptors.Device.bNumConfigurations == 0) continue;

			PUSB_CONFIGURATION Config = &Descriptors.Config;
			//printf("%d: bNumInterfaces %d\r\n", i, Config->bNumInterfaces);
			if (Config->bNumInterfaces == 0) continue;
			PUSB_INTERFACE Interface = &Descriptors.Interface;
			//printf("%d: Interface class %d,%d,%d (3,1,1)\r\n", i, Interface->bInterfaceClass, Interface->bInterfaceSubClass, Interface->bInterfaceProtocol);
			if (Interface->bInterfaceClass != USB_CLASS_HID) continue;
			if (Interface->bInterfaceSubClass != USB_SUBCLASS_BOOT) continue;
			if (Interface->bInterfaceProtocol != USB_PROTOCOL_KEYBOARD) continue;

			for (ULONG Ep = 0; Ep < Interface->bNumEndpoints; Ep++) {
				PUSB_ENDPOINT Endpoint = &Descriptors.Endpoints[Ep];
				//printf("%d(%d): %d,%02x (3,bit7)\r\n", i, Ep, Endpoint->bmAttributes, Endpoint->bEndpointAddress);

				if (Endpoint->bmAttributes != USB_ENDPOINT_INTERRUPT) continue;
				if ((Endpoint->bEndpointAddress & USB_ENDPOINT_IN) == 0) continue;
				s_UsbHandle = DeviceHandle;
				Configuration = Config->bConfigurationValue;
				s_Interface = Interface->bInterfaceNumber;
				AltInterface = Interface->bAlternateSetting;
				s_Endpoint = Endpoint->bEndpointAddress;
				MaxPacketSize = Endpoint->wMaxPacketSize;
				//VendorId = Entries[i].VendorId;
				//ProductId = Entries[i].ProductId;
				Found = true;
				break;
			}
		} while (0);
		if (!Found) UlCloseDevice(DeviceHandle);
	}
	PxiIopFree(Entries);
	if (!Found) return -1;

	//CHAR DebugText[512];
	UCHAR CurrentConf = 0;
	LONG Status = UlGetConfiguration(s_UsbHandle, &CurrentConf);

	if (Status < 0) {
#if 0
		_snprintf(DebugText, sizeof(DebugText), "UlGetConfiguration returned %08x\n", Status);
		HalDisplayString(DebugText);
		while (1);
#endif
		//printf("UlGetConfiguration returned %d\r\n", Status);
		return Status;
	}

	if (CurrentConf != Configuration) {
		Status = UlSetConfiguration(s_UsbHandle, Configuration);
		if (Status < 0) {
#if 0
			_snprintf(DebugText, sizeof(DebugText), "UlSetConfiguration returned %08x\n", Status);
			HalDisplayString(DebugText);
			while (1);
#endif
			//printf("UlSetConfiguration returned %d\r\n", Status);
			return Status;
		}
	}

	if (AltInterface != 0) {
		Status = UlSetAlternativeInterface(s_UsbHandle, s_Interface, AltInterface);
		if (Status < 0) {
#if 0
			_snprintf(DebugText, sizeof(DebugText), "UlSetAlternativeInterface returned %08x\n", Status);
			HalDisplayString(DebugText);
			while (1);
#endif
			//printf("UlSetAlternativeInterface returned %d\r\n", Status);
			return Status;
		}
	}

	UCHAR Protocol;
	Status = UlkpGetProtocol(&Protocol);
	if (Status < 0 || Protocol != 0) {
#if 0
		if (Status < 0) {
			_snprintf(DebugText, sizeof(DebugText), "UlkpGetProtocol returned %08x\n", Status);
			HalDisplayString(DebugText);
		}
#endif
		//printf("UlkpGetProtocol returned %d\r\n", Status);
		Status = UlkpSetProtocol(0);
		if (Status < 0) {
#if 0
			_snprintf(DebugText, sizeof(DebugText), "UlkpSetProtocol returned %08x\n", Status);
			HalDisplayString(DebugText);
			while (1);
#endif
			return Status;
		}
		Status = UlkpGetProtocol(&Protocol);
		if (Status < 0 && Protocol == 1) {
			//printf("UlkpGetProtocol returned %d\r\n", Status);
			return -1;
		}
	}

	// Allocate memory for write report
	s_ReportWrite = PxiIopAlloc(sizeof(*s_ReportWrite));
	if (s_ReportWrite == NULL) {
		return -1;
	}

	// Allocate memory for read report
	s_ReportRead = PxiIopAlloc(sizeof(*s_ReportRead));
	if (s_ReportRead == NULL) {
		PxiIopFree(s_ReportWrite);
		s_ReportWrite = NULL;
		return -1;
	}

	// Make sure keyboard leds are all off
	//UlkSetLeds(0);

	// Fire off the first async receive.
	Status = UlkpStartReceiveInputReport();
	if (Status < 0) {
		printf("UlkpStartReceiveInputReport returned %d\r\n", Status);
		PxiIopFree(s_ReportRead);
		s_ReportRead = NULL;
	}
	return Status;
}