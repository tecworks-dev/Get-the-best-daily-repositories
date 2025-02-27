#pragma once

typedef struct _USB_MOUSE_REPORT {
	UCHAR Buttons;
	signed char X;
	signed char Y;
	signed char Wheel;
	UCHAR Padding[0x1C];
} USB_MOUSE_REPORT, *PUSB_MOUSE_REPORT;
_Static_assert(sizeof(USB_MOUSE_REPORT) == 0x20);

// Mouse device object.
extern PDEVICE_OBJECT MouDeviceObject;

// Called from lower-level driver when mouse input data read complete.
void MouReadComplete(PUSB_MOUSE_REPORT Report, UCHAR Length);

// USB mouse ioctl implementation
NTSTATUS MouIoctl(PDEVICE_OBJECT Device, PIRP Irp);

// Initialise USB mouse driver.
NTSTATUS MouInit(PDRIVER_OBJECT Driver, PUNICODE_STRING RegistryPath);