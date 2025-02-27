// USB keyboard driver over IOS usblow
#define DEVL 1
#include <ntddk.h>
#include "keyboard.h"
#include "usblow.h"
#include "usblowkbd.h"
#include "iosapi.h"
#include "asynctimer.h"

#include <stdio.h>

static IOS_USB_HANDLE s_UsbHandle;
static PUSB_KBD_REPORT s_ReportRead = NULL;
static PU8_AS_32 s_ReportWrite = NULL;
//static HANDLE s_ReportReadThread = NULL;
static KTIMER s_KbdTimer;
static KDPC s_KbdDpc;
static UCHAR s_Interface;
static UCHAR s_Endpoint;

static void UlkpReceivedInputReport(NTSTATUS Status, ULONG Value, PVOID Context);

static NTSTATUS UlkpStartReceiveInputReport(void) {
	return UlTransferInterruptMessageAsyncDpc(
		s_UsbHandle,
		s_Endpoint,
		sizeof(*s_ReportRead),
		s_ReportRead,
		UlkpReceivedInputReport,
		NULL
	);
}

static void UlkpTimerCallback(PKDPC Dpc, PVOID DeferredContext, PVOID SystemArgument1, PVOID SystemArgument2) {
	NTSTATUS Status = UlkpStartReceiveInputReport();
	
	if (!NT_SUCCESS(Status)) {
		AsyncTimerSet(&s_KbdTimer, &s_KbdDpc);
	}
}

static void UlkpReceivedInputReport(NTSTATUS Status, ULONG Value, PVOID Context) {
	Context = UlGetPassedAsyncContext(Context);
	
	if (NT_SUCCESS(Status)) {
		KbdReadComplete(s_ReportRead);
	}
	
	UlkpTimerCallback(NULL, NULL, NULL, NULL);
}

NTSTATUS UlkSetLeds(UCHAR UsbIndicators) {
	if (s_ReportWrite == NULL) return STATUS_NO_SUCH_DEVICE;
	
	U8_AS_32 Usb32 = {.Long = 0};
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

static NTSTATUS UlkpGetProtocol(PUCHAR Protocol) {
	PUCHAR Buffer = HalIopAlloc(sizeof(*Protocol));
	if (Buffer == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	NTSTATUS Status = UlTransferControlMessage(
		s_UsbHandle,
		USB_REQTYPE_INTERFACE_GET,
		USB_REQ_GETPROTOCOL,
		0,
		s_Interface,
		1,
		Buffer
	);
	if (NT_SUCCESS(Status)) *Protocol = *Buffer;
	HalIopFree(Buffer);
	return Status;
}

static NTSTATUS UlkpSetProtocol(UCHAR Protocol) {
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

NTSTATUS UlkInit(void) {
	if (s_ReportRead != NULL) return STATUS_SUCCESS;
	
	PIOS_USB_DEVICE_ENTRY Entries = (PIOS_USB_DEVICE_ENTRY)
		HalIopAlloc(sizeof(IOS_USB_DEVICE_ENTRY_MAX));
	if (Entries == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	RtlZeroMemory(Entries, sizeof(IOS_USB_DEVICE_ENTRY_MAX));
	
	UCHAR NumHid;
	UlGetDeviceList(Entries, USB_COUNT_DEVICES, USB_CLASS_HID, &NumHid);
	
	BOOLEAN Found = FALSE;
	UCHAR Configuration, AltInterface;
	USHORT MaxPacketSize;
	
	//CHAR DebugText[512];
	
#if 0
		_snprintf(DebugText, sizeof(DebugText), "USBHID devices count: %d\n", NumHid);
		HalDisplayString(DebugText);
		if (NumHid == 0) while(1);
#endif
	
	for (ULONG i = 0; !Found && i < NumHid; i++) {
		if (Entries[i].VendorId == 0 || Entries[i].ProductId == 0) {
			continue;
		}
		
#if 0
		_snprintf(DebugText, sizeof(DebugText), "%d: %04x:%04x\n", i, Entries[i].VendorId, Entries[i].ProductId);
		HalDisplayString(DebugText);
#endif
		
		IOS_USB_HANDLE DeviceHandle = Entries[i].DeviceHandle;
		NTSTATUS Status = UlOpenDevice(DeviceHandle);
		if (!NT_SUCCESS(Status)) {
#if 0
			_snprintf(DebugText, sizeof(DebugText), "%d: UlOpenDevice failed %08x\n", i, Status);
			HalDisplayString(DebugText);
#endif
			continue;
		}
		USB_DEVICE_DESC Descriptors;
		Status = UlGetDescriptors(DeviceHandle, &Descriptors);
		if (!NT_SUCCESS(Status)) {
#if 0
			_snprintf(DebugText, sizeof(DebugText), "%d: UlGetDescriptors failed %08x\n", i, Status);
			HalDisplayString(DebugText);
#endif
			continue;
		}
		
#if 0
		_snprintf(DebugText, sizeof(DebugText), "%d: bNumConfigurations=%d\n", i, Descriptors.Device.bNumConfigurations);
		HalDisplayString(DebugText);
#endif
		if (Descriptors.Device.bNumConfigurations == 0) continue;
		
		PUSB_CONFIGURATION Config = &Descriptors.Config;
		
#if 0
		_snprintf(DebugText, sizeof(DebugText), "%d: bNumInterfaces=%d\n", i, Config->bNumInterfaces);
		HalDisplayString(DebugText);
#endif
		if (Config->bNumInterfaces == 0) continue;
		PUSB_INTERFACE Interface = &Descriptors.Interface;
		
#if 0
		_snprintf(DebugText, sizeof(DebugText), "%d: Interface class %d,%d,%d (3,1,1)\r\n", i, Interface->bInterfaceClass, Interface->bInterfaceSubClass, Interface->bInterfaceProtocol);
		HalDisplayString(DebugText);
#endif
		if (Interface->bInterfaceClass != USB_CLASS_HID) continue;
		if (Interface->bInterfaceSubClass != USB_SUBCLASS_BOOT) continue;
		if (Interface->bInterfaceProtocol != USB_PROTOCOL_KEYBOARD) continue;
		
		for (ULONG Ep = 0; Ep < Interface->bNumEndpoints; Ep++) {
			PUSB_ENDPOINT Endpoint = &Descriptors.Endpoints[Ep];
#if 0
			_snprintf(DebugText, sizeof(DebugText), "%d(%d): %d,%02x (3,bit7)\r\n", i, Ep, Endpoint->bmAttributes, Endpoint->bEndpointAddress);
			HalDisplayString(DebugText);
#endif
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
			Found = TRUE;
			break;
		}
		if (!Found) UlCloseDevice(DeviceHandle);
	}
	HalIopFree(Entries);
	if (!Found) {
#if 0
		while (1);
#endif
		return STATUS_NO_SUCH_DEVICE;
	}
	
	//CHAR DebugText[512];
	UCHAR CurrentConf = 0;
	NTSTATUS Status = UlGetConfiguration(s_UsbHandle, &CurrentConf);
	
	if (!NT_SUCCESS(Status)) {
#if 0
		_snprintf(DebugText, sizeof(DebugText), "UlGetConfiguration returned %08x\n", Status);
		HalDisplayString(DebugText);
		while (1);
#endif
		return Status;
	}
	
	if (CurrentConf != Configuration) {
		Status = UlSetConfiguration(s_UsbHandle, Configuration);
		if (!NT_SUCCESS(Status)) {
#if 0
			_snprintf(DebugText, sizeof(DebugText), "UlSetConfiguration returned %08x\n", Status);
			HalDisplayString(DebugText);
			while (1);
#endif
			return Status;
		}
	}
	
	if (AltInterface != 0) {
		Status = UlSetAlternativeInterface(s_UsbHandle, s_Interface, AltInterface);
		if (!NT_SUCCESS(Status)) {
#if 0
			_snprintf(DebugText, sizeof(DebugText), "UlSetAlternativeInterface returned %08x\n", Status);
			HalDisplayString(DebugText);
			while (1);
#endif
			return Status;
		}
	}
		
	UCHAR Protocol;
	Status = UlkpGetProtocol(&Protocol);
	if (!NT_SUCCESS(Status) || Protocol != 0) {
#if 0
		if (!NT_SUCCESS(Status)) {
			_snprintf(DebugText, sizeof(DebugText), "UlkpGetProtocol returned %08x\n", Status);
			HalDisplayString(DebugText);
		}
#endif
		Status = UlkpSetProtocol(0);
		if (!NT_SUCCESS(Status)) {
#if 0
			_snprintf(DebugText, sizeof(DebugText), "UlkpSetProtocol returned %08x\n", Status);
			HalDisplayString(DebugText);
			while (1);
#endif
			return Status;
		}
		Status = UlkpGetProtocol(&Protocol);
		if (NT_SUCCESS(Status) && Protocol == 1) {
			return STATUS_NO_SUCH_DEVICE;
		}
	}
	
	// Allocate memory for write report
	s_ReportWrite = HalIopAlloc(sizeof(*s_ReportWrite));
	if (s_ReportWrite == NULL) {
		return STATUS_INSUFFICIENT_RESOURCES;
	}
	
	// Allocate memory for read report
	s_ReportRead = HalIopAlloc(sizeof(*s_ReportRead));
	if (s_ReportRead == NULL) {
		HalIopFree(s_ReportWrite);
		s_ReportWrite = NULL;
		return STATUS_INSUFFICIENT_RESOURCES;
	}
	
	// Attempt to ensure all keyboard LEDs are off.
	UlkSetLeds(0);
	
	// Initialise the timer and DPC.
	KeInitializeDpc(&s_KbdDpc, UlkpTimerCallback, NULL);
	KeInitializeTimer(&s_KbdTimer);
	Status = UlkpStartReceiveInputReport();
	if (!NT_SUCCESS(Status)) {
		HalIopFree(s_ReportRead);
		s_ReportRead = NULL;
		HalIopFree(s_ReportWrite);
		s_ReportWrite = NULL;
	}
	return Status;
}