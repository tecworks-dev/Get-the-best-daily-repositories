// USB stack driver (entrypoint)
#define DEVL 1
#include <ntddk.h>
#include "keyboard.h"
#include "mouse.h"
#include "usbkbd.h"
#include "usblow.h"
#include "usblowkbd.h"
#include "usblowmou.h"
#include "usblowms.h"

NTSTATUS UsbCreate(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
	// just return success
	Irp->IoStatus.Status = STATUS_SUCCESS;
	IoCompleteRequest(Irp, IO_NO_INCREMENT);
	return STATUS_SUCCESS;
}

NTSTATUS UsbInternalDeviceControl(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
	if (DeviceObject == KbdDeviceObject) return KbdIoctl(DeviceObject, Irp);
	if (DeviceObject == MouDeviceObject) return MouIoctl(DeviceObject, Irp);
	Irp->IoStatus.Status = STATUS_INVALID_DEVICE_REQUEST;
	IoCompleteRequest(Irp, IO_NO_INCREMENT);
	return STATUS_INVALID_DEVICE_REQUEST;
}

NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
	// Attempt to initialise the IOS usb/kbd driver.
	// If this is present, then that is the ONLY IOS usb component we can use.
	BOOLEAN UsingKbd = FALSE;
	BOOLEAN UsingMouse = FALSE;
	BOOLEAN UsingLow = FALSE;
	NTSTATUS Status = IukInit();
	if (NT_SUCCESS(Status)) {
		UsingKbd = TRUE;
	} else {
		// Initialise usb low-level driver.
		//HalDisplayString("UlInit\n");
		Status = UlInit();
		if (!NT_SUCCESS(Status)) {
			// IOS USB keyboard and USB low-level v5 drivers failed to init.
			// Can't do anything...
			return STATUS_NO_SUCH_DEVICE;
		}
		
		UsingLow = TRUE;
		// Initialise usb-low keyboard and mouse drivers.
		//HalDisplayString("UlkInit\n");
		Status = UlkInit();
		if (NT_SUCCESS(Status)) UsingKbd = TRUE;
		//HalDisplayString("UlmInit\n");
		Status = UlmInit();
		if (NT_SUCCESS(Status)) UsingMouse = TRUE;
	}
	
	// Got usb keyboard or mouse or both, so create devices.
	if (UsingKbd) {
		//HalDisplayString("KbdInit\n");
		Status = KbdInit(DriverObject, RegistryPath);
		if (!NT_SUCCESS(Status)) UsingKbd = FALSE;
	}
	if (UsingMouse) {
		//HalDisplayString("MouInit\n");
		Status = MouInit(DriverObject, RegistryPath);
		if (!NT_SUCCESS(Status)) UsingMouse = FALSE;
	}
	
	if (!UsingKbd && !UsingMouse && !UsingLow) {
		// Initialisation failed such that this driver will perform no operation.
		// We have to keep this driver up, return success.
		return STATUS_SUCCESS;
	}
	
	// Set up the device driver entry points.
	DriverObject->MajorFunction[IRP_MJ_CREATE] = UsbCreate;
	DriverObject->MajorFunction[IRP_MJ_CLOSE] = UsbCreate;
	DriverObject->MajorFunction[IRP_MJ_INTERNAL_DEVICE_CONTROL] = UsbInternalDeviceControl;
	
	// Attempt to init mass storage via usb low.
	if (UsingLow) {
		//HalDisplayString("UlmsInit\n");
		UlmsInit(DriverObject);
	}
	return STATUS_SUCCESS;
}