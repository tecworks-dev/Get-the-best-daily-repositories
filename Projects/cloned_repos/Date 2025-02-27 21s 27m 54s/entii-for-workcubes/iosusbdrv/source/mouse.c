// Mouse driver NT side.
// Includes conversion from USB HID packet to what mouclass expects.
#define DEVL 1
#include <ntddk.h>
#include <kbdmou.h>
#include <stdio.h>
#include "mouse.h"
#define SIZEOF_ARRAY(x) (sizeof((x)) / sizeof((x)[0]))
#define RtlCopyMemory(Destination,Source,Length) memcpy((Destination),(Source),(Length))

// Button conversion tables from mouhid.sys
static const USHORT sc_UsbToPs2ButtonUp[] = {
	MOUSE_BUTTON_1_UP,
	MOUSE_BUTTON_2_UP,
	MOUSE_BUTTON_3_UP,
	//MOUSE_BUTTON_4_UP,
	//MOUSE_BUTTON_5_UP
};
static const USHORT sc_UsbToPs2ButtonDown[] = {
	MOUSE_BUTTON_1_DOWN,
	MOUSE_BUTTON_2_DOWN,
	MOUSE_BUTTON_3_DOWN,
	//MOUSE_BUTTON_4_DOWN,
	//MOUSE_BUTTON_5_DOWN
};

PDEVICE_OBJECT MouDeviceObject = NULL;
static CONNECT_DATA s_ClassConnection = {0};
static ULONG s_EnableCount = 0;
static MOUSE_INPUT_DATA s_InputData = {0};
static UCHAR s_OldButtons = 0;

static BOOLEAN s_CreatedOneMouseDevice = FALSE;

void MouReadComplete(PUSB_MOUSE_REPORT Report, UCHAR Length) {
	if (!s_EnableCount) return;
	if (s_ClassConnection.ClassService == NULL) return;
	// We have a single input report.
	// Convert to the format that mouclass accepts.
	// If mouclass hasn't initialised yet; or mouclass is disabled; don't bother.
	if (s_EnableCount == 0 || s_ClassConnection.ClassService == NULL) return;
	PMOUSE_INPUT_DATA InputDataEnd = &s_InputData;
	InputDataEnd++;
	// Buttons first.
	UCHAR Buttons = Report->Buttons;
	UCHAR ButtonsDelta = Buttons ^ s_OldButtons;
	s_InputData.ButtonFlags = 0;
	for (int Bit = 0; Bit < 3; Bit++, ButtonsDelta >>= 1, Buttons >>= 1) {
		if ((ButtonsDelta & 1) == 0) continue;
		USHORT Flag = 0;
		if ((Buttons & 1) != 0) {
			Flag = sc_UsbToPs2ButtonDown[Bit];
		} else {
			Flag = sc_UsbToPs2ButtonUp[Bit];
		}
		s_InputData.ButtonFlags |= Flag;
	}
	s_OldButtons = Report->Buttons;
	// X/Y/Wheel movement.
	s_InputData.LastX = Report->X;
	s_InputData.LastY = Report->Y;
	if (Length > __builtin_offsetof(USB_MOUSE_REPORT, Wheel)) {
		s_InputData.ButtonData = Report->Wheel;
		if (Report->Wheel != 0) {
			s_InputData.ButtonFlags |= MOUSE_WHEEL;
		}
	}
	
	PSERVICE_CALLBACK_ROUTINE OnMouseInput = (PSERVICE_CALLBACK_ROUTINE)
		s_ClassConnection.ClassService;
	// Callback needs to be called at DISPATCH_LEVEL
	KIRQL OldIrql;
	KeRaiseIrql(DISPATCH_LEVEL, &OldIrql);
	ULONG ElementsRead;
	OnMouseInput(s_ClassConnection.ClassDeviceObject, &s_InputData, InputDataEnd, &ElementsRead);
	KeLowerIrql(OldIrql);
	if (ElementsRead != 1) {
		// should never happen
	}
}

static NTSTATUS MoupIoctl(PDEVICE_OBJECT Device, PIRP Irp) {
	// assumption: Device == MouDeviceObject
	PIO_STACK_LOCATION Stack = IoGetCurrentIrpStackLocation(Irp);
	__auto_type Params = &Stack->Parameters.DeviceIoControl;
	
	switch (Params->IoControlCode) {
	case IOCTL_INTERNAL_MOUSE_CONNECT:
		// Mouse class driver is giving us its device object and callback.
		if (s_ClassConnection.ClassService != NULL) {
			// ...but it already did!
			return STATUS_SHARING_VIOLATION;
		}
		if (Params->InputBufferLength < sizeof(s_ClassConnection)) {
			// ...but the buffer was too small
			return STATUS_INVALID_PARAMETER;
		}
		RtlCopyMemory(&s_ClassConnection, Params->Type3InputBuffer, sizeof(s_ClassConnection));
		return STATUS_SUCCESS;
	case IOCTL_INTERNAL_MOUSE_DISCONNECT:
		// Mouse class driver is telling us to stop using its callback.
		// But nobody ever implements this, so why should we bother?
		return STATUS_NOT_IMPLEMENTED;
	case IOCTL_INTERNAL_MOUSE_ENABLE:
		// Mouse class driver is telling us to increment the enable count.
		if (s_EnableCount == 0xFFFFFFFF) return STATUS_DEVICE_DATA_ERROR;
		InterlockedIncrement(&s_EnableCount);
		return STATUS_SUCCESS;
	case IOCTL_INTERNAL_MOUSE_DISABLE:
		// Mouse class driver is telling us to decrement the enable count.
		if (s_EnableCount == 0) return STATUS_DEVICE_DATA_ERROR;
		InterlockedDecrement(&s_EnableCount);
		return STATUS_SUCCESS;
	case IOCTL_MOUSE_QUERY_ATTRIBUTES:
		// Caller wants mouse attributes which are basically hardcoded.
		if (Params->OutputBufferLength < sizeof(MOUSE_ATTRIBUTES)) {
			return STATUS_BUFFER_TOO_SMALL;
		}
		{
			PMOUSE_ATTRIBUTES Attributes = (PMOUSE_ATTRIBUTES)
				Irp->AssociatedIrp.SystemBuffer;
			RtlZeroMemory(Attributes, sizeof(*Attributes));
			Attributes->SampleRate = 0;
			Attributes->InputDataQueueLength = 2;
			// Claim to have a middle button always.
			// If no middle button actually exists, that bit will just never be set.
			Attributes->NumberOfButtons = 3;
			// Additionally, always claim to have a wheel.
			// Again, if no wheel actually exists, that value will always equal zero, that is, no movement.
			Attributes->MouseIdentifier = WHEELMOUSE_I8042_HARDWARE;
		}
		return STATUS_SUCCESS;
	}
	return STATUS_INVALID_DEVICE_REQUEST;
}

NTSTATUS MouIoctl(PDEVICE_OBJECT Device, PIRP Irp) {
	NTSTATUS Status = MoupIoctl(Device, Irp);
	Irp->IoStatus.Status = Status;
	IoCompleteRequest(Irp, IO_NO_INCREMENT);
	return Status;
}

NTSTATUS MouInit(PDRIVER_OBJECT Driver, PUNICODE_STRING RegistryPath) {
	// Create only one pointer device.
	if (MouDeviceObject != NULL) return STATUS_SHARING_VIOLATION;
	// We expect no other pointer devices to have been created;
	// but just in case, loop through all of them anyways.
	NTSTATUS Status = STATUS_UNSUCCESSFUL;
	for (ULONG i = 0; i < POINTER_PORTS_MAXIMUM; i++) {
		// Generate the NT object path name for this device.
		UCHAR NtName[256];
		sprintf(NtName, DD_POINTER_PORT_DEVICE_NAME "%d", i);
		// Wrap an ANSI string around it.
		STRING NtNameStr;
		RtlInitString(&NtNameStr, NtName);
		// Convert to unicode.
		UNICODE_STRING NtNameUs;
		Status = RtlAnsiStringToUnicodeString(&NtNameUs, &NtNameStr, TRUE);
		if (!NT_SUCCESS(Status)) continue;
		// Create the device object.
		Status = IoCreateDevice(Driver, 0, &NtNameUs, FILE_DEVICE_8042_PORT, 0, FALSE, &MouDeviceObject);
		if (!NT_SUCCESS(Status)) {
			RtlFreeUnicodeString(&NtNameUs);
			continue;
		}
		// Created it.
		// if a USB mouse is plugged in prioritise that over an SI device
		//if (i != 0) s_CreatedOneMouseDevice = TRUE;
		if (!s_CreatedOneMouseDevice) {
			static USHORT s_MousePortKey[] = {
				'P', 'o', 'i', 'n', 't', 'e', 'r',
				'P', 'o', 'r', 't', 0
			};
			// Write the registry device map, mouclass expects to see this.
			Status = RtlWriteRegistryValue(
				RTL_REGISTRY_DEVICEMAP,
				s_MousePortKey,
				NtNameUs.Buffer,
				REG_SZ,
				RegistryPath->Buffer,
				RegistryPath->Length
			);
			if (NT_SUCCESS(Status)) s_CreatedOneMouseDevice = TRUE;
		}
		// Free the unicode string.
		RtlFreeUnicodeString(&NtNameUs);
		break;
	}
	return Status;
}