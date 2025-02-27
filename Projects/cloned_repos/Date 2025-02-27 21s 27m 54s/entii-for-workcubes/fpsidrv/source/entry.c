// SI driver (entrypoint)
#define DEVL 1
#include <ntddk.h>
#include "si.h"
#include "keyboard.h"
#include "mouse.h"

NTSTATUS SiCreate(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
	// just return success
	Irp->IoStatus.Status = STATUS_SUCCESS;
	IoCompleteRequest(Irp, IO_NO_INCREMENT);
	return STATUS_SUCCESS;
}

NTSTATUS SiInternalDeviceControl(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
	if (DeviceObject == KbdDeviceObject) return KbdIoctl(DeviceObject, Irp);
	if (DeviceObject == MouDeviceObject) return MouIoctl(DeviceObject, Irp);
	Irp->IoStatus.Status = STATUS_INVALID_DEVICE_REQUEST;
	IoCompleteRequest(Irp, IO_NO_INCREMENT);
	return STATUS_INVALID_DEVICE_REQUEST;
}

BOOLEAN SikbdInit(void);

NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
	NTSTATUS Status = SiInit();
	if (!NT_SUCCESS(Status)) return Status;
	
	if (!SikbdInit()) return STATUS_SUCCESS;
	KbdInit(DriverObject, RegistryPath);
	MouInit(DriverObject, RegistryPath);
	
	// Set up the device driver entry points.
	DriverObject->MajorFunction[IRP_MJ_CREATE] = SiCreate;
	DriverObject->MajorFunction[IRP_MJ_CLOSE] = SiCreate;
	DriverObject->MajorFunction[IRP_MJ_INTERNAL_DEVICE_CONTROL] = SiInternalDeviceControl;
	
	return STATUS_SUCCESS;
}