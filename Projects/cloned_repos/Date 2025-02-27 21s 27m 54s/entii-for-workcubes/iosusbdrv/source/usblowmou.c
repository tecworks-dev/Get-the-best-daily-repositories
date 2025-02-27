// USB mouse support via usblow
#define DEVL 1
#include <ntddk.h>
#include "usblowmou.h"
#include "mouse.h"
#include "usblow.h"
#include "iosapi.h"
#include "asynctimer.h"

static IOS_USB_HANDLE s_UsbHandle;
static PUSB_MOUSE_REPORT s_ReportRead = NULL;
//static HANDLE s_ReportReadThread = NULL;
static KTIMER s_MouTimer;
static KDPC s_MouDpc;
static UCHAR s_Interface;
static UCHAR s_Endpoint;
static USHORT s_EndpointSize;

static void UlmpReceivedInputReport(NTSTATUS Status, ULONG Value, PVOID Context);

static NTSTATUS UlmpStartReceiveInputReport(void) {
	return UlTransferInterruptMessageAsyncDpc(
		s_UsbHandle,
		s_Endpoint,
		s_EndpointSize,
		s_ReportRead,
		UlmpReceivedInputReport,
		NULL
	);
}

static void UlmpTimerCallback(PKDPC Dpc, PVOID DeferredContext, PVOID SystemArgument1, PVOID SystemArgument2) {
	NTSTATUS Status = UlmpStartReceiveInputReport();
	
	if (!NT_SUCCESS(Status)) {
		AsyncTimerSet(&s_MouTimer, &s_MouDpc);
	}
}


static void UlmpReceivedInputReport(NTSTATUS Status, ULONG Value, PVOID Context) {
	Context = UlGetPassedAsyncContext(Context);
	
	if (NT_SUCCESS(Status)) {
		MouReadComplete(s_ReportRead, Value);
	}
	
	UlmpTimerCallback(NULL, NULL, NULL, NULL);
}

static NTSTATUS UlmpSetProtocol(UCHAR Protocol) {
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

NTSTATUS UlmInit(void) {
	if (s_ReportRead != NULL) return STATUS_SUCCESS;
	
	PIOS_USB_DEVICE_ENTRY Entries = (PIOS_USB_DEVICE_ENTRY)
		HalIopAlloc(sizeof(IOS_USB_DEVICE_ENTRY_MAX));
	if (Entries == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	RtlZeroMemory(Entries, sizeof(IOS_USB_DEVICE_ENTRY_MAX));
	
	UCHAR NumHid;
	UlGetDeviceList(Entries, USB_COUNT_DEVICES, USB_CLASS_HID, &NumHid);
	
	//CHAR DebugText[512];
	
#if 0
		_snprintf(DebugText, sizeof(DebugText), "USBHID devices count: %d\n", NumHid);
		HalDisplayString(DebugText);
		//if (NumHid == 0) while(1);
#endif
	
	BOOLEAN Found = FALSE;
	UCHAR Configuration, AltInterface;
	USHORT MaxPacketSize;
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
		if (!NT_SUCCESS(UlGetDescriptors(DeviceHandle, &Descriptors))) {
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
		_snprintf(DebugText, sizeof(DebugText), "%d: Interface class %d,%d,%d (3,1,2)\r\n", i, Interface->bInterfaceClass, Interface->bInterfaceSubClass, Interface->bInterfaceProtocol);
		HalDisplayString(DebugText);
#endif
		if (Interface->bInterfaceClass != USB_CLASS_HID) continue;
		if (Interface->bInterfaceSubClass != USB_SUBCLASS_BOOT) continue;
		if (Interface->bInterfaceProtocol != USB_PROTOCOL_MOUSE) continue;
		
		for (ULONG Ep = 0; Ep < Interface->bNumEndpoints; Ep++) {
			PUSB_ENDPOINT Endpoint = &Descriptors.Endpoints[Ep];
#if 0
			_snprintf(DebugText, sizeof(DebugText), "%d(%d): %d,%02x,%x (3,bit7,20)\r\n", i, Ep, Endpoint->bmAttributes, Endpoint->bEndpointAddress, Endpoint->wMaxPacketSize);
			HalDisplayString(DebugText);
#endif
			if (Endpoint->bmAttributes != USB_ENDPOINT_INTERRUPT) continue;
			if ((Endpoint->bEndpointAddress & USB_ENDPOINT_IN) == 0) continue;
			if (Endpoint->wMaxPacketSize > sizeof(USB_MOUSE_REPORT)) continue;
			s_UsbHandle = DeviceHandle;
			Configuration = Config->bConfigurationValue;
			s_Interface = Interface->bInterfaceNumber;
			AltInterface = Interface->bAlternateSetting;
			s_Endpoint = Endpoint->bEndpointAddress;
			s_EndpointSize = Endpoint->wMaxPacketSize;
			Found = TRUE;
			break;
		}
		if (!Found) UlCloseDevice(DeviceHandle);
	}
	HalIopFree(Entries);
	if (!Found) return STATUS_NO_SUCH_DEVICE;
	
	// Set boot protocol
	NTSTATUS Status = UlmpSetProtocol(0);
	if (!NT_SUCCESS(Status)) {
#if 0
		_snprintf(DebugText, sizeof(DebugText), "UlmpSetProtocol returned %08x\n", Status);
		HalDisplayString(DebugText);
		while (1);
#endif
		return Status;
	}
	
	// Allocate memory for read report
	s_ReportRead = HalIopAlloc(sizeof(*s_ReportRead));
	if (s_ReportRead == NULL) {
		return STATUS_INSUFFICIENT_RESOURCES;
	}
	
	// Initialise the timer and DPC.
	KeInitializeDpc(&s_MouDpc, UlmpTimerCallback, NULL);
	KeInitializeTimer(&s_MouTimer);
	// Start receiving input reports.
	Status = UlmpStartReceiveInputReport();
	if (!NT_SUCCESS(Status)) {
#if 0
		_snprintf(DebugText, sizeof(DebugText), "UlmpStartReceiveInputReport returned %08x\n", Status);
		HalDisplayString(DebugText);
		while (1);
#endif
		HalIopFree(s_ReportRead);
		s_ReportRead = NULL;
	}
	return Status;
}