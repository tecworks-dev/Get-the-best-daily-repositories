// USB mass storage support via usblow
// Despite USB mass storage being SCSI, we will not be using NT scsi drivers.
#define DEVL 1
#include <ntddk.h>
#include <hal.h>
#include <ntstatus.h>
#include <ntddcdrm.h>
#include <ntdddisk.h>
#include <ntddscsi.h>
#include <stdio.h>
#include "usblowms.h"
#include "usblow.h"
#include "usbms.h"
#include "iosapi.h"
#include "zwdd.h"

#define SIGNATURE_PARTITION 'UPRT'
#define SIGNATURE_DISK 'UDSK'

typedef enum {
	USBMS_DISK_UNKNOWN,
	USBMS_DISK_FLOPPY,
	USBMS_DISK_CDROM,
	USBMS_DISK_OTHER_FIXED,
	USBMS_DISK_OTHER_REMOVABLE
} USBMS_DISK_TYPE, *PUSBMS_DISK_TYPE;

typedef struct _USBMS_EXTENSION USBMS_EXTENSION, *PUSBMS_EXTENSION;

typedef struct _USBMS_PARTITION {
	ULONG Signature;
	struct _USBMS_PARTITION * NextPartition;
	PUSBMS_EXTENSION PhysicalDisk;
	LARGE_INTEGER PartitionLength;
	LARGE_INTEGER StartingOffset;
	// Extra stuff needed for PARTITION_INFORMATION
	ULONG HiddenSectors;
	ULONG PartitionNumber;
	UCHAR PartitionType;
} USBMS_PARTITION, *PUSBMS_PARTITION;

typedef struct _USBMS_CONTROLLER {
	IOS_USB_HANDLE DeviceHandle;
	PVOID Buffer;
	ULONG MaxSize;
	UCHAR EndpointIn;
	UCHAR EndpointOut;
	UCHAR Interface;
	UCHAR CbwTag;
} USBMS_CONTROLLER, *PUSBMS_CONTROLLER;

struct _USBMS_EXTENSION {
	ULONG Signature;
	// Workaround for changer.sys bug and probably some other stupid drivers too:
	// changer.sys expects to see a scsiclass DEVICE_EXTENSION on any FILE_DEVICE_CD_ROM device.
	// and expects to see a device object at ->PortDeviceObject (offset 4) there.
	// so, put our device object pointer here, so it doesn't try to use something else as a device object!
	PDEVICE_OBJECT DeviceObject;
	PCONTROLLER_OBJECT Controller;
	ULONG SectorSize;
	ULONG SectorShift;
	ULONG SectorCount;
	BOOLEAN WriteProtected;
	BOOLEAN DriveNotReady;
	UCHAR Lun;
	ULONG DiskNumber;
	PUSBMS_PARTITION NextPartition;
	KDPC FinishDpc;
};

typedef struct _USBMS_WORK_ITEM {
	WORK_QUEUE_ITEM WorkItem;
	PDEVICE_OBJECT DeviceObject;
	PIRP Irp;
	PUSBMS_EXTENSION ExtDisk;
} USBMS_WORK_ITEM, *PUSBMS_WORK_ITEM;

// PARTITION_INFORMATION should be 64 bits aligned
_Static_assert(sizeof(PARTITION_INFORMATION) == 0x20);

// USB timeout stuff:
// assumption: as a mass storage device, nothing's spinlooping waiting on us
// BUGBUG: any timeouting IPC request *will* leak memory :(
// but it'll also leak memory on the IOS side, so...
//static ULONG s_SecondsTimeout = USBSTORAGE_TIMEOUT;
#define MS_TO_TIMEOUT(ms) ((ms) * 10000)

// USB controller ARC path.
// Must be synchronised with arcfw\source\arcdisk.c
static char s_ControllerPath[] = "scsi(0)";

NTSTATUS MspCtrlMsgTimeout(IOS_USB_HANDLE DeviceHandle, UCHAR RequestType, UCHAR Request, USHORT Value, USHORT Index, USHORT Length, PVOID Data, ULONG SecondsTimeout) {
	PIOS_USB_ASYNC_RESULT Async = ExAllocatePool(NonPagedPool, sizeof(IOS_USB_ASYNC_RESULT));
	if (Async == NULL) return STATUS_NO_MEMORY;
	
	NTSTATUS Status = UlTransferControlMessageAsync(DeviceHandle, RequestType, Request, Value, Index, Length, Data, Async, NULL);
	if (!NT_SUCCESS(Status)) {
		ExFreePool(Async);
		return Status;
	}
	
	// Wait on the event with timeout.
	LARGE_INTEGER Timeout = {.QuadPart = 0};
	Timeout.QuadPart = SecondsTimeout * 1000;
	Timeout.QuadPart = -MS_TO_TIMEOUT(Timeout.QuadPart);
	Status = KeWaitForSingleObject(&Async->Event, Executive, KernelMode, FALSE, &Timeout);
	
	if (Status == STATUS_TIMEOUT) {
		// Cancel the control message.
		UlCancelEndpoint(DeviceHandle, USB_CANCEL_CONTROL);
		// And wait for the failure IPC response.
		KeWaitForSingleObject(&Async->Event, Executive, KernelMode, FALSE, NULL);
		// Turn it into a failing status.
		Status = STATUS_IO_TIMEOUT;
	}
	if (NT_SUCCESS(Status)) {
		// Got an IPC response in time.
		Status = Async->Status;
	}
	
	// Ensure the usblow context gets freed.
	UlGetPassedAsyncContext( Async->Buffer );
	
	// Free the async context.
	ExFreePool( Async );
	
	// And return.
	return Status;
}

NTSTATUS MspBulkMsgTimeout(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data, ULONG SecondsTimeout) {
	PIOS_USB_ASYNC_RESULT Async = ExAllocatePool(NonPagedPool, sizeof(IOS_USB_ASYNC_RESULT));
	if (Async == NULL) return STATUS_NO_MEMORY;
	
	NTSTATUS Status = UlTransferBulkMessageAsync(DeviceHandle, Endpoint, Length, Data, Async, NULL);
	if (!NT_SUCCESS(Status)) {
		ExFreePool(Async);
		return Status;
	}
	
	// Wait on the event with timeout.
	LARGE_INTEGER Timeout = {.QuadPart = 0};
	Timeout.QuadPart = SecondsTimeout * 1000;
	Timeout.QuadPart = -MS_TO_TIMEOUT(Timeout.QuadPart);
	Status = KeWaitForSingleObject(&Async->Event, Executive, KernelMode, FALSE, &Timeout);
	
	if (Status == STATUS_TIMEOUT) {
		// Cancel the endpoint message.
		UlCancelEndpoint(DeviceHandle, Endpoint);
		// And wait for the failure IPC response.
		KeWaitForSingleObject(&Async->Event, Executive, KernelMode, FALSE, NULL);
		// Turn it into a failing status.
		Status = STATUS_IO_TIMEOUT;
	}
	
	if (NT_SUCCESS(Status)) {
		// Got an IPC response in time.
		Status = Async->Status;
	}
	
	// Ensure the usblow context gets freed.
	UlGetPassedAsyncContext( Async->Buffer );
	
	// Free the async context.
	ExFreePool( Async );
	
	// And return.
	return Status;
}

NTSTATUS MspReset(IOS_USB_HANDLE DeviceHandle, UCHAR Interface, UCHAR EndpointIn, UCHAR EndpointOut) {
	NTSTATUS Status = MspCtrlMsgTimeout(
		DeviceHandle,
		(USB_CTRLTYPE_DIR_HOST2DEVICE | USB_CTRLTYPE_TYPE_CLASS | USB_CTRLTYPE_REC_INTERFACE),
		USBSTORAGE_RESET,
		0,
		Interface,
		0,
		NULL,
		1
	);
	UlCancelEndpoint(DeviceHandle, EndpointIn);
	UlCancelEndpoint(DeviceHandle, EndpointOut);
	UlClearHalt(DeviceHandle);
	return Status;
}

NTSTATUS MspSendCbw(
	PUSBMS_CONTROLLER Controller,
	UCHAR Lun,
	ULONG Length,
	UCHAR Flags,
	PUCHAR Cb,
	UCHAR CbLength,
	ULONG SecondsTimeout,
	PULONG pTag
) {
	if (CbLength == 0 || CbLength > 16) return STATUS_INVALID_PARAMETER;
	
	USBMS_CBW Cbw;
	RtlZeroMemory(&Cbw, sizeof(Cbw));
	PUSBMS_CBW pCbw = HalIopAlloc( sizeof(USBMS_CBW) );
	if (pCbw == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	Cbw.Signature = CBW_SIGNATURE;
	Controller->CbwTag++;
	*pTag = Controller->CbwTag;
	Cbw.Tag = Controller->CbwTag;
	Cbw.Length = Length;
	Cbw.Flags = Flags;
	Cbw.CbLength = (CbLength > 6 ? 10 : 6);
	RtlCopyMemory(Cbw.Cb, Cb, CbLength);
	RtlCopyMemory(pCbw, &Cbw, sizeof(Cbw));
	
	NTSTATUS Status = MspBulkMsgTimeout(
		Controller->DeviceHandle,
		Controller->EndpointOut,
		CBW_SIZE,
		pCbw,
		SecondsTimeout
	);
	HalIopFree(pCbw);
	
	if (!NT_SUCCESS(Status)) return Status;
	
	LONG IopResult = STATUS_TO_IOP(Status);
	if (IopResult == CBW_SIZE) return STATUS_SUCCESS;
	return STATUS_IO_DEVICE_ERROR;
}

NTSTATUS MspReadCsw(
	PUSBMS_CONTROLLER Controller,
	ULONG Tag,
	PUCHAR CswStatus,
	PULONG DataResidue,
	ULONG SecondsTimeout
) {
	PUSBMS_CSW Csw = HalIopAlloc(sizeof(USBMS_CSW));
	if (Csw == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	RtlZeroMemory(Csw, sizeof(*Csw));
	NTSTATUS Status = MspBulkMsgTimeout(
		Controller->DeviceHandle,
		Controller->EndpointIn,
		CSW_SIZE,
		Csw,
		SecondsTimeout
	);
	
	do {
		if (!NT_SUCCESS(Status)) break;
		LONG IopResult = STATUS_TO_IOP(Status);
		if (IopResult != CSW_SIZE) {
			Status = STATUS_IO_DEVICE_ERROR;
			break;
		}
		
		if (Csw->Signature != CSW_SIGNATURE) {
			Status = STATUS_DEVICE_PROTOCOL_ERROR;
			break;
		}
		Status = STATUS_SUCCESS;
		if (CswStatus != NULL) {
			*CswStatus = Csw->Status;
		}
		if (DataResidue != NULL) {
			*DataResidue = Csw->DataResidue;
		}
		
		if (Csw->Tag != Tag) {
			Status = STATUS_DEVICE_PROTOCOL_ERROR;
		}
	} while (FALSE);
	
	HalIopFree(Csw);
	return Status;
}

NTSTATUS MspRunScsi(
	PUSBMS_CONTROLLER Controller,
	UCHAR Lun,
	PUCHAR Buffer,
	ULONG Length,
	PUCHAR Cb,
	UCHAR CbLength,
	BOOLEAN Write,
	PUCHAR CswStatus,
	PULONG CswDataResidue,
	ULONG SecondsTimeout
) {
	// Toggle the disc LED gpio.
	MmioWrite32((PVOID)0x8d8000c0, MmioRead32((PVOID)0x8d8000c0) ^ 0x20);
	USHORT MaxSize = Controller->MaxSize;
	NTSTATUS Status = STATUS_SUCCESS;
	ULONG Endpoint = (Write ? Controller->EndpointOut : Controller->EndpointIn );
	UCHAR LocalCswStatus = 0;
	ULONG LocalDataResidue = 0;
	for (UCHAR Retry = 0; Retry < USBSTORAGE_CYCLE_RETRIES; Retry++) {
		ULONG CurrLength = Length;
		PUCHAR Pointer = Buffer;
		if (Status == STATUS_IO_TIMEOUT) break;
		ULONG Tag;
		
		Status = MspSendCbw(
			Controller, Lun,
			CurrLength,
			( Write ? CBW_OUT : CBW_IN ),
			Cb,
			CbLength,
			SecondsTimeout,
			&Tag
		);
		
		while (CurrLength > 0 && NT_SUCCESS(Status)) {
			ULONG RoundLength = CurrLength > MaxSize ? MaxSize : CurrLength;
			
			// Always go through the map buffer.
			if (Write && Buffer != NULL) RtlCopyMemory( Controller->Buffer, Pointer, RoundLength );
			Status = MspBulkMsgTimeout(
				Controller->DeviceHandle,
				Endpoint,
				RoundLength,
				Controller->Buffer,
				SecondsTimeout
			);
			ULONG BytesTransferred = 0;
			if (NT_SUCCESS(Status)) {
				BytesTransferred = STATUS_TO_IOP( Status );
				if (!Write && Buffer != NULL) {
					RtlCopyMemory( Pointer, Controller->Buffer, BytesTransferred );
					HalSweepDcacheRange(Pointer, BytesTransferred);
				}
				if (BytesTransferred == RoundLength) {
					CurrLength -= BytesTransferred;
					if (Buffer != NULL) Pointer += BytesTransferred;
				}
				else Status = STATUS_IO_DEVICE_ERROR;
			}
		}
		
		if (NT_SUCCESS(Status)) {
			Status = MspReadCsw(Controller, Tag, &LocalCswStatus, &LocalDataResidue, SecondsTimeout);
		}
		
		if (!NT_SUCCESS(Status)) {
			NTSTATUS ResetStatus = MspReset(
				Controller->DeviceHandle,
				Controller->Interface,
				Controller->EndpointIn,
				Controller->EndpointOut
			);
			if (ResetStatus == STATUS_IO_TIMEOUT) Status = ResetStatus;
		} else {
			Status = STATUS_SUCCESS;
			break;
		}
	}
	
	if (CswStatus != NULL) *CswStatus = LocalCswStatus;
	if (CswDataResidue != NULL) *CswDataResidue = LocalDataResidue;
	// Toggle the disc LED gpio.
	MmioWrite32((PVOID)0x8d8000c0, MmioRead32((PVOID)0x8d8000c0) ^ 0x20);
	return Status;
}

NTSTATUS MspRunScsiMeta(
	PUSBMS_CONTROLLER Controller,
	UCHAR Lun,
	PUCHAR Buffer,
	ULONG Length,
	PUCHAR Cb,
	UCHAR CbLength,
	PULONG TransferredLength,
	PUCHAR CswStatus,
	PULONG CswDataResidue,
	ULONG SecondsTimeout
) {
	USHORT MaxSize = Controller->MaxSize;
	if (Length > MaxSize) return STATUS_INVALID_PARAMETER;
	// This is a metadata request.
	// Instead of using the buffer from the controller;
	// allocate a map buffer from the IPC area.
	// This is so ioctls can be performed synchronously.
	PUCHAR MapBuffer = NULL;
	if (Length != 0) {
		MapBuffer = (PUCHAR) HalIopAlloc(Length);
		if (MapBuffer == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	}
	NTSTATUS Status = STATUS_SUCCESS;
	ULONG Endpoint = Controller->EndpointIn;
	UCHAR LocalCswStatus = 0;
	ULONG LocalDataResidue = 0;
	for (UCHAR Retry = 0; Retry < USBSTORAGE_CYCLE_RETRIES; Retry++) {
		if (Status == STATUS_IO_TIMEOUT) break;
		ULONG Tag;
		
		Status = MspSendCbw(
			Controller, Lun,
			Length,
			CBW_IN,
			Cb,
			CbLength,
			SecondsTimeout,
			&Tag
		);
		
		if (NT_SUCCESS(Status) && Length != 0) {
			// Always go through the map buffer.
			Status = MspBulkMsgTimeout(
				Controller->DeviceHandle,
				Endpoint,
				Length,
				MapBuffer,
				SecondsTimeout
			);
			ULONG BytesTransferred = 0;
			if (NT_SUCCESS(Status)) {
				if (Buffer != NULL) {
					BytesTransferred = STATUS_TO_IOP( Status );
					RtlCopyMemory( Buffer, MapBuffer, BytesTransferred );
					HalSweepDcacheRange(Buffer, BytesTransferred);
				}
				if (TransferredLength) *TransferredLength = BytesTransferred;
			}
		}
		
		if (NT_SUCCESS(Status)) {
			Status = MspReadCsw(Controller, Tag, &LocalCswStatus, &LocalDataResidue, SecondsTimeout);
		}
		
		if (!NT_SUCCESS(Status)) {
			NTSTATUS ResetStatus = MspReset(
				Controller->DeviceHandle,
				Controller->Interface,
				Controller->EndpointIn,
				Controller->EndpointOut
			);
			if (ResetStatus == STATUS_IO_TIMEOUT) Status = ResetStatus;
		} else {
			Status = STATUS_SUCCESS;
			break;
		}
	}
	
	if (CswStatus != NULL) *CswStatus = LocalCswStatus;
	if (CswDataResidue != NULL) *CswDataResidue = LocalDataResidue;
	if (MapBuffer != NULL) HalIopFree(MapBuffer);
	return Status;
}

NTSTATUS MspLowRead10(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, USHORT SectorCount, ULONG SectorShift, PVOID Buffer) {
	NTSTATUS Status;
	UCHAR CswStatus;
	UCHAR ScsiCdb[] = {
		SCSI_READ_10,
		Lun << 5,
		SectorStart >> 24,
		SectorStart >> 16,
		SectorStart >> 8,
		SectorStart >> 0,
		0,
		SectorCount >> 8,
		SectorCount >> 0,
		0
	};
	ULONG Length = SectorCount << SectorShift;
	
	// Use a 10 seconds timeout to allow for drive to wake up if needed
	Status = MspRunScsi(
		Controller,
		Lun,
		Buffer,
		Length,
		ScsiCdb,
		sizeof(ScsiCdb),
		FALSE,
		&CswStatus,
		NULL,
		10
	);
	if (!NT_SUCCESS(Status)) return Status;
	if (CswStatus != 0) return STATUS_IO_DEVICE_ERROR;
	return STATUS_SUCCESS;
}

NTSTATUS MspLowWrite10(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, USHORT SectorCount, ULONG SectorShift, PVOID Buffer) {
	NTSTATUS Status;
	UCHAR CswStatus;
	UCHAR ScsiCdb[] = {
		SCSI_WRITE_10,
		Lun << 5,
		SectorStart >> 24,
		SectorStart >> 16,
		SectorStart >> 8,
		SectorStart >> 0,
		0,
		SectorCount >> 8,
		SectorCount >> 0,
		0
	};
	ULONG Length = SectorCount << SectorShift;
	
	// Use a 10 seconds timeout to allow for drive to wake up if needed
	Status = MspRunScsi(
		Controller,
		Lun,
		Buffer,
		Length,
		ScsiCdb,
		sizeof(ScsiCdb),
		TRUE,
		&CswStatus,
		NULL,
		10
	);
	if (!NT_SUCCESS(Status)) return Status;
	if (CswStatus != 0) return STATUS_IO_DEVICE_ERROR;
	return STATUS_SUCCESS;
}

NTSTATUS MspLowRead(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, ULONG SectorCount, ULONG SectorShift, PUCHAR Buffer) {
	USHORT CurrentRound;
	NTSTATUS Status;
	while (SectorCount != 0) {
		if (SectorCount > 0xFFFF) CurrentRound = 0xFFFF;
		else CurrentRound = (USHORT)SectorCount;
		
		Status = MspLowRead10(Controller, Lun, SectorStart, CurrentRound, SectorShift, Buffer);
		if (!NT_SUCCESS(Status)) return Status;
		
		ULONG Length = CurrentRound << SectorShift;
		SectorStart += CurrentRound;
		SectorCount -= CurrentRound;
		Buffer += Length;
	}
	return STATUS_SUCCESS;
}

NTSTATUS MspLowVerify(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, ULONG SectorCount, ULONG SectorShift) {
	USHORT CurrentRound;
	NTSTATUS Status;
	while (SectorCount != 0) {
		if (SectorCount > 0xFFFF) CurrentRound = 0xFFFF;
		else CurrentRound = (USHORT)SectorCount;
		
		Status = MspLowRead10(Controller, Lun, SectorStart, CurrentRound, SectorShift, NULL);
		if (!NT_SUCCESS(Status)) return Status;
		
		ULONG Length = CurrentRound << SectorShift;
		SectorStart += CurrentRound;
		SectorCount -= CurrentRound;
	}
	return STATUS_SUCCESS;
}

NTSTATUS MspLowWrite(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, ULONG SectorCount, ULONG SectorShift, PUCHAR Buffer) {
	USHORT CurrentRound;
	NTSTATUS Status;
	while (SectorCount != 0) {
		if (SectorCount > 0xFFFF) CurrentRound = 0xFFFF;
		else CurrentRound = (USHORT)SectorCount;
		
		Status = MspLowWrite10(Controller, Lun, SectorStart, CurrentRound, SectorShift, Buffer);
		if (!NT_SUCCESS(Status)) return Status;
		
		ULONG Length = CurrentRound << SectorShift;
		SectorStart += CurrentRound;
		SectorCount -= CurrentRound;
		Buffer += Length;
	}
	return STATUS_SUCCESS;
}


NTSTATUS MspClearErrors(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SecondsTimeout) {
	NTSTATUS Status;
	UCHAR ScsiCdb[6];
	UCHAR Sense[SCSI_SENSE_REPLY_SIZE];
	UCHAR CswStatus = 0;
		
	RtlZeroMemory(&ScsiCdb, sizeof(ScsiCdb));
	ScsiCdb[0] = SCSI_TEST_UNIT_READY;
	
	Status = MspRunScsiMeta(Controller, Lun, NULL, 0, ScsiCdb, sizeof(ScsiCdb), NULL, &CswStatus, NULL, SecondsTimeout);
	if (!NT_SUCCESS(Status)) return Status;
	
	if (CswStatus) {
		ScsiCdb[0] = SCSI_REQUEST_SENSE;
		ScsiCdb[1] = Lun << 5;
		ScsiCdb[4] = sizeof(Sense);
		ULONG Transferred = 0;
		Status = MspRunScsiMeta(Controller, Lun, Sense, sizeof(Sense), ScsiCdb, sizeof(ScsiCdb), &Transferred, NULL, NULL, SecondsTimeout);
		if (!NT_SUCCESS(Status)) return Status;
		if (Transferred < sizeof(Sense)) return STATUS_DEVICE_PROTOCOL_ERROR;
		
		UCHAR SenseErr = Sense[2] & 0xF;
		if (SenseErr == SCSI_SENSE_NOT_READY) return STATUS_DEVICE_NOT_READY;
		if (SenseErr == SCSI_SENSE_MEDIUM_ERROR) return STATUS_DEVICE_DATA_ERROR;
		if (SenseErr == SCSI_SENSE_HARDWARE_ERROR) return STATUS_IO_DEVICE_ERROR;
		if (SenseErr == SCSI_SENSE_UNIT_ATTENTION) return STATUS_VERIFY_REQUIRED;
	}
	
	return Status;
}

BOOLEAN MspCapacityIsFloppy(PSCSI_FORMAT_CAPACITY_ENTRY Entries, UCHAR Count) {
	for (ULONG i = 0; i < Count; i++) {
		ULONG NumberOfBlocks = Entries[i].NumberOfBlocks;
		ULONG BlockLength = (Entries[i].BlockLengthHigh << 16) |
			Entries[i].BlockLengthLow;
		if (BlockLength == 1024) {
			if (NumberOfBlocks == (77 * 2 * 8)) {
				// 1.25MB (PC-98, etc)
				return TRUE;
			}
			continue;
		}
		if (BlockLength != 512) continue;
		// for 512-byte sectors, allow everything that NT5 usbmass.sys does;
		// assume that usbmass.sys supports it because some USB floppy drive does.
		switch (NumberOfBlocks) {
			case  80 *  2 * 8:  // 640 KB (5.25"?!)
			case  80 *  2 * 9:  // 720 KB
			case  80 *  2 * 15: // 1.2 MB
			case  80 *  2 * 18: // 1.44MB
			
			case 963 *  8 * 32: // 120MB, SuperDisk LS-120
			
			case 890 * 13 * 34: // 200MB, Sony HiFD
				return TRUE;
		}
	}
	return FALSE;
}

NTSTATUS MspGetDiskType(PUSBMS_CONTROLLER Controller, UCHAR Lun, PUSBMS_DISK_TYPE DiskType, ULONG SecondsTimeout) {
	// Do SCSI Inquiry, get 36 bytes
	NTSTATUS Status;
	UCHAR Inquiry[36];
	UCHAR ScsiCdb[6] = {0};
	ScsiCdb[0] = SCSI_INQUIRY;
	ScsiCdb[1] = Lun << 5;
	ScsiCdb[4] = sizeof(Inquiry);
	for (int Retry = 0; Retry < 2; Retry++) {
		RtlZeroMemory(Inquiry, sizeof(Inquiry));
		ULONG Transferred = 0;
		Status = MspRunScsiMeta(Controller, Lun, Inquiry, sizeof(Inquiry), ScsiCdb, sizeof(ScsiCdb), &Transferred, NULL, NULL, SecondsTimeout);
		if (!NT_SUCCESS(Status)) continue;
		if (Transferred < sizeof(Inquiry)) {
			Status = STATUS_DEVICE_PROTOCOL_ERROR;
			continue;
		}
		break;
	}
	if (!NT_SUCCESS(Status)) return Status;
	
	BOOLEAN RemovableDisk = (Inquiry[1] & 0x80) != 0;
	UCHAR DeviceType = Inquiry[0] & 0x1f;
	if (DeviceType == 5) {
		*DiskType = USBMS_DISK_CDROM;
		return STATUS_SUCCESS;
	}
	if (DeviceType != 0) {
		*DiskType = USBMS_DISK_UNKNOWN;
		return STATUS_SUCCESS;
	}
	// is this a floppy drive?
	// assume non-floppy until it is known to be otherwise
	*DiskType = RemovableDisk ? USBMS_DISK_OTHER_REMOVABLE : USBMS_DISK_OTHER_FIXED;
	// send SCSI Read Format Capacities to find out
	// if any entry matches a known floppy size;
	// then this is a floppy disk
	UCHAR FormatCapsBuf[
		sizeof(SCSI_FORMAT_CAPACITY_LIST) +
		(31 * sizeof(SCSI_FORMAT_CAPACITY_ENTRY))
	];
	UCHAR ScsiCdb12[12];
	RtlZeroMemory(FormatCapsBuf, sizeof(FormatCapsBuf));
	RtlZeroMemory(ScsiCdb12, sizeof(ScsiCdb12));
	ScsiCdb12[0] = SCSI_READ_FORMAT_CAPACITY;
	ScsiCdb12[1] = Lun << 5;
	ScsiCdb12[8] = sizeof(FormatCapsBuf);
	ULONG TransferredBytes = 0;
	Status = MspRunScsiMeta(Controller, Lun, FormatCapsBuf, sizeof(FormatCapsBuf), ScsiCdb12, sizeof(ScsiCdb12), &TransferredBytes, NULL, NULL, SecondsTimeout);
	if (!NT_SUCCESS(Status)) return STATUS_SUCCESS;
	if (TransferredBytes < sizeof(SCSI_FORMAT_CAPACITY_LIST)) return STATUS_SUCCESS;
	PSCSI_FORMAT_CAPACITY_LIST List = (PSCSI_FORMAT_CAPACITY_LIST)
		FormatCapsBuf;
	if (List->Length == 0) return STATUS_SUCCESS;
	if ((List->Length % sizeof(SCSI_FORMAT_CAPACITY_ENTRY)) != 0) return STATUS_SUCCESS;
	
	ULONG Count = TransferredBytes - sizeof(SCSI_FORMAT_CAPACITY_ENTRY);
	if (Count >= List->Length) Count = List->Length;
	Count /= sizeof(SCSI_FORMAT_CAPACITY_ENTRY);
	
	if (MspCapacityIsFloppy(List->Entries, Count)) {
		*DiskType = USBMS_DISK_FLOPPY;
	}
	
	return STATUS_SUCCESS;
}

NTSTATUS MspIsWriteProtect(PUSBMS_CONTROLLER Controller, UCHAR Lun, PBOOLEAN WriteProtect, ULONG SecondsTimeout) {
	NTSTATUS Status;
	UCHAR ScsiCdb6[6] = {0};
	ScsiCdb6[0] = SCSI_MODE_SENSE;
	ScsiCdb6[2] = 0x3F;
	ScsiCdb6[4] = 192;
	UCHAR ModeSense[192];
	
	ULONG Transferred = 0;
	Status = MspRunScsiMeta(Controller, Lun, ModeSense, sizeof(ModeSense), ScsiCdb6, sizeof(ScsiCdb6), &Transferred, NULL, NULL, SecondsTimeout);
	
	if (!NT_SUCCESS(Status)) return Status;
	
	if (Transferred < 4) {
		// try again
		Status = MspRunScsiMeta(Controller, Lun, ModeSense, sizeof(ModeSense), ScsiCdb6, sizeof(ScsiCdb6), &Transferred, NULL, NULL, SecondsTimeout);
		
		if (!NT_SUCCESS(Status)) return Status;
		if (Transferred < 4) return STATUS_IO_DEVICE_ERROR;
	}
	
	*WriteProtect = (ModeSense[2] & 0x80) != 0;
	return STATUS_SUCCESS;
}

NTSTATUS MspReadCapacity(PUSBMS_CONTROLLER Controller, UCHAR Lun, PULONG SectorSize, PULONG SectorCount, ULONG SecondsTimeout) {
	NTSTATUS Status;
	UCHAR ScsiCdb10[10] = {0};
	ScsiCdb10[0] = SCSI_READ_CAPACITY;
	ScsiCdb10[1] = (Lun << 5);
	SCSI_READ_CAPACITY_RES ReadCapacity;
	
	ULONG Transferred = 0;
	Status = MspRunScsiMeta(Controller, Lun, (PUCHAR)&ReadCapacity, sizeof(ReadCapacity), ScsiCdb10, sizeof(ScsiCdb10), &Transferred, NULL, NULL, SecondsTimeout);
	if (!NT_SUCCESS(Status)) return Status;
	
	if (SectorCount != NULL) *SectorCount = ReadCapacity.SectorCount;
	if (SectorSize != NULL) *SectorSize = ReadCapacity.SectorSize;
	return STATUS_SUCCESS;
}

NTSTATUS MspCheckVerify(PDEVICE_OBJECT DeviceObject, PUSBMS_CONTROLLER Controller, PUSBMS_EXTENSION Ext) {
	// Do clear errors. Use 20 second timeout just in case.
	UCHAR Lun = Ext->Lun;
	NTSTATUS Status = MspClearErrors(Controller, Lun, 20);
	NTSTATUS CapStatus;
	BOOLEAN IsCdRom = DeviceObject->DeviceType == FILE_DEVICE_CD_ROM;
	if (Status == STATUS_VERIFY_REQUIRED) {
		// Medium changed. Go again, allow media to spin up, ignore result.
		Status = MspClearErrors(Controller, Lun, 20);
		// Read the drive capacity.
		ULONG SectorCount = 0;
		ULONG SectorSize = 0;
		NTSTATUS CapStatus = MspReadCapacity(Controller, Lun, &SectorSize, &SectorCount, 20);
		Ext->DriveNotReady = !NT_SUCCESS(CapStatus);
		if (!NT_SUCCESS(CapStatus) || SectorSize == 0) {
			if (IsCdRom) {
				SectorCount = 0x7fffffff;
				SectorSize = 2048;
			} else {
				SectorCount = 0;
				SectorSize = 512;
			}
		}
		// set the new sector-count/size
		Ext->SectorSize = SectorSize;
		Ext->SectorCount = SectorCount;
		Ext->SectorShift = __builtin_ctz(SectorSize);
	}
	// Run mode sense to get write protect status.
	BOOLEAN WriteProtect = IsCdRom;
	if (!WriteProtect) {
		CapStatus = MspIsWriteProtect(Controller, Lun, &WriteProtect, USBSTORAGE_TIMEOUT);
		if (!NT_SUCCESS(CapStatus)) WriteProtect = FALSE;
	}
	Ext->WriteProtected = WriteProtect;
	return Status;
}

static void MspFinishDpc(
	PKDPC Dpc,
	PVOID DeferredContext,
	PVOID SystemArgument1,
	PVOID SystemArgument2
) {
	// Get the device object and the IRP.
	PDEVICE_OBJECT DeviceObject = (PDEVICE_OBJECT)DeferredContext;
	PIRP Irp = (PIRP)SystemArgument1;
	PCONTROLLER_OBJECT Controller = (PCONTROLLER_OBJECT)SystemArgument2;
	
	// Complete the request.
	IoCompleteRequest(Irp, IO_DISK_INCREMENT);
	// Release the controller object.
	IoFreeController(Controller);
	// Start next packet.
	IoStartNextPacket(DeviceObject, FALSE);
}

static NTSTATUS MspDeviceCreateImpl(
	PDRIVER_OBJECT DriverObject,
	PCONTROLLER_OBJECT Controller,
	UCHAR Lun,
	ULONG SectorSize,
	ULONG SectorCount,
	PULONG DeviceCount,
	ULONG ArcKey,
	DEVICE_TYPE DeviceType,
	ULONG DeviceCharacteristics,
	const char * DevicePath
) {
	BOOLEAN IsHardDisk = DeviceType == FILE_DEVICE_DISK && ((DeviceCharacteristics & FILE_FLOPPY_DISKETTE) == 0);
	// Generate the NT object path name for this device.
	UCHAR NtName[256];
	sprintf(NtName, "\\Device\\%s%d", DevicePath, *DeviceCount);
	
	// Wrap an ANSI string around it.
	STRING NtNameStr;
	RtlInitString(&NtNameStr, NtName);
	
	// Convert to unicode.
	UNICODE_STRING NtNameUs;
	NTSTATUS Status = RtlAnsiStringToUnicodeString(&NtNameUs, &NtNameStr, TRUE);
	if (!NT_SUCCESS(Status)) return Status;

	HANDLE hDir = NULL;
	if (IsHardDisk) {
		// For hard disks, we create a directory, and Partition%d objects inside that
		// (where Partition0 means the entire disk)
		// We also don't need to symlink the arc name (kernel will do that for us)
		OBJECT_ATTRIBUTES Oa;
		InitializeObjectAttributes(&Oa, &NtNameUs, OBJ_CASE_INSENSITIVE | OBJ_PERMANENT, NULL, NULL);
		Status = ZwCreateDirectoryObject(&hDir, DIRECTORY_ALL_ACCESS, &Oa);
		if (!NT_SUCCESS(Status)) return Status;
		RtlFreeUnicodeString(&NtNameUs);
		// Generate the NT object path name for the actual device.
		sprintf(NtName, "\\Device\\%s%d\\Partition0", DevicePath, *DeviceCount);
		RtlInitString(&NtNameStr, NtName);
		Status = RtlAnsiStringToUnicodeString(&NtNameUs, &NtNameStr, TRUE);
		// Let the following code create the device object.
	}
	
	PDEVICE_OBJECT DeviceObject = NULL;
	do {
		if (!NT_SUCCESS(Status)) break;
		// Create the device object.
		Status = IoCreateDevice(DriverObject, sizeof(USBMS_EXTENSION), &NtNameUs, DeviceType, DeviceCharacteristics, FALSE, &DeviceObject);
		if (!NT_SUCCESS(Status)) break;
	
		// Symlink it to the arc name, if needed.
		if (!IsHardDisk) {
			UCHAR ArcName[256];
			sprintf(ArcName,
				"\\ArcName\\%s%s(%u)fdisk(0)",
				s_ControllerPath,
				DeviceType == FILE_DEVICE_CD_ROM ? "cdrom" : "disk",
				ArcKey
			);
			STRING ArcNameStr;
			RtlInitString(&ArcNameStr, ArcName);
			UNICODE_STRING ArcNameUs;
			Status = RtlAnsiStringToUnicodeString(&ArcNameUs, &ArcNameStr, TRUE);
			if (!NT_SUCCESS(Status)) {
				RtlFreeUnicodeString(&NtNameUs);
				break;
			}
			// This might fail if there were multiple LUNs.
			// ARC firmware only uses the first one anyway, so don't bother checking.
			IoAssignArcName(&ArcNameUs, &NtNameUs);
			RtlFreeUnicodeString(&ArcNameUs);
		} else {
			// We don't need the directory handle any more.
			ZwClose(hDir);
			hDir = NULL;
		}
		RtlFreeUnicodeString(&NtNameUs);
		
		// Initialise the new device object.
		DeviceObject->Flags |= DO_DIRECT_IO;
		DeviceObject->AlignmentRequirement = 32;
		DeviceObject->StackSize = (DeviceType == FILE_DEVICE_CD_ROM ? 2 : 1);
		
		// Initialise the device extension.
		PUSBMS_EXTENSION Ext = (PUSBMS_EXTENSION) DeviceObject->DeviceExtension;
		Ext->Controller = Controller;
		Ext->Lun = Lun;
		Ext->WriteProtected = (DeviceCharacteristics & FILE_READ_ONLY_DEVICE) != 0;
		Ext->Signature = SIGNATURE_DISK;
		Ext->DeviceObject = DeviceObject;
		Ext->DiskNumber = *DeviceCount;
		Ext->SectorSize = SectorSize;
		Ext->SectorCount = SectorCount;
		Ext->SectorShift = __builtin_ctz(SectorSize);
		Ext->DriveNotReady = FALSE;
		KeInitializeDpc(&Ext->FinishDpc, MspFinishDpc, DeviceObject);
		// TODO: more needed here?
		
		// For a hard disk, create all partition objects.
		if (IsHardDisk) {
			PDRIVE_LAYOUT_INFORMATION PartitionList;
			Status = IoReadPartitionTable(DeviceObject, SectorSize, TRUE, (PVOID)&PartitionList);
			if (NT_SUCCESS(Status)) {
				PUSBMS_PARTITION ExtPart = NULL;
				for (ULONG Partition = 0; Partition < PartitionList->PartitionCount; Partition++) {
					sprintf(NtName, "\\Device\\%s%d\\Partition%d", DevicePath, *DeviceCount, Partition + 1);
					RtlInitString(&NtNameStr, NtName);
					Status = RtlAnsiStringToUnicodeString(&NtNameUs, &NtNameStr, TRUE);
					if (!NT_SUCCESS(Status)) return Status;
					Status = IoCreateDevice(DriverObject, sizeof(USBMS_PARTITION), &NtNameUs, DeviceType, DeviceCharacteristics, FALSE, &DeviceObject);
					RtlFreeUnicodeString(&NtNameUs);
					if (!NT_SUCCESS(Status)) return Status;
					
					DeviceObject->Flags |= DO_DIRECT_IO;
					DeviceObject->AlignmentRequirement = 32;
					DeviceObject->StackSize = 1;
					
					if (ExtPart == NULL) {
						Ext->NextPartition = (PUSBMS_PARTITION) DeviceObject->DeviceExtension;
						ExtPart = Ext->NextPartition;
					} else {
						ExtPart->NextPartition = (PUSBMS_PARTITION) DeviceObject->DeviceExtension;
						ExtPart = ExtPart->NextPartition;
					}
					ExtPart->PhysicalDisk = Ext;
					PPARTITION_INFORMATION PartitionInfo = &PartitionList->PartitionEntry[Partition];
					ExtPart->PartitionLength.QuadPart = PartitionInfo->PartitionLength.QuadPart;
					ExtPart->StartingOffset.QuadPart = PartitionInfo->StartingOffset.QuadPart;
					ExtPart->PartitionType = PartitionInfo->PartitionType;
					ExtPart->PartitionNumber = PartitionInfo->PartitionNumber;
					ExtPart->HiddenSectors = PartitionInfo->HiddenSectors;
					ExtPart->Signature = SIGNATURE_PARTITION;
				}
				ExFreePool(PartitionList);
			}
		}
		
		// All done.
		return Status;
	} while (FALSE);
	
	if (hDir != NULL) {
		// Delete the directory.
		ZwMakeTemporaryObject(hDir);
		ZwClose(hDir);
		RtlFreeUnicodeString(&NtNameUs);
	}
	if (DeviceObject != NULL) IoDeleteDevice(DeviceObject);
	return Status;
}

NTSTATUS MspDeviceCreate(
	PDRIVER_OBJECT DriverObject,
	PCONTROLLER_OBJECT Controller,
	UCHAR Lun,
	ULONG SectorSize,
	ULONG SectorCount,
	PULONG DeviceCount,
	ULONG ArcKey,
	DEVICE_TYPE DeviceType,
	ULONG DeviceCharacteristics,
	const char * DevicePath
) {
	NTSTATUS Status = MspDeviceCreateImpl(
		DriverObject,
		Controller,
		Lun,
		SectorSize,
		SectorCount,
		DeviceCount,
		ArcKey,
		DeviceType,
		DeviceCharacteristics,
		DevicePath
	);
	if (!NT_SUCCESS(Status)) return Status;
	*DeviceCount += 1;
	return Status;
}

NTSTATUS MspDiskCreate(
	PDRIVER_OBJECT DriverObject,
	PCONTROLLER_OBJECT Controller,
	UCHAR Lun,
	ULONG SectorSize,
	ULONG SectorCount,
	PCONFIGURATION_INFORMATION Config,
	ULONG ArcKey,
	USBMS_DISK_TYPE DiskType
) {
	if (DiskType == USBMS_DISK_UNKNOWN) return STATUS_UNSUCCESSFUL;
	DEVICE_TYPE DeviceType;
	ULONG DeviceCharacteristics;
	const char * DevicePath = NULL;
	PULONG DeviceCount = NULL;
	
	if (DiskType == USBMS_DISK_FLOPPY) {
		DeviceType = FILE_DEVICE_DISK;
		DeviceCharacteristics = FILE_REMOVABLE_MEDIA | FILE_FLOPPY_DISKETTE;
		DevicePath = "Floppy";
		DeviceCount = &Config->FloppyCount;
	} else if (DiskType == USBMS_DISK_CDROM) {
		DeviceType = FILE_DEVICE_CD_ROM;
		DeviceCharacteristics = FILE_REMOVABLE_MEDIA | FILE_READ_ONLY_DEVICE;
		DevicePath = "CdRom";
		DeviceCount = &Config->CdRomCount;
	} else if (DiskType == USBMS_DISK_OTHER_FIXED || DiskType == USBMS_DISK_OTHER_REMOVABLE) {
		DeviceType = FILE_DEVICE_DISK;
		DeviceCharacteristics = 
#if 0 // OTHER_REMOVABLE is probably flash storage so mount it as fixed
			DiskType == USBMS_DISK_OTHER_REMOVABLE ?
			FILE_REMOVABLE_MEDIA : 0;
#else
			0;
#endif
		DevicePath = "Harddisk";
		DeviceCount = &Config->DiskCount;
	} else {
		return STATUS_UNSUCCESSFUL;
	}
	
	return MspDeviceCreate(
		DriverObject,
		Controller,
		Lun,
		SectorSize,
		SectorCount,
		DeviceCount,
		ArcKey,
		DeviceType,
		DeviceCharacteristics,
		DevicePath
	);
}

// Read/write handler
NTSTATUS MsDiskRw(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
	PUSBMS_EXTENSION Ext = (PUSBMS_EXTENSION)DeviceObject->DeviceExtension;
	// Ensure the device extension is actually valid.
	if (Ext == NULL || (
		Ext->Signature != SIGNATURE_PARTITION &&
		Ext->Signature != SIGNATURE_DISK
	)) {
		// this is probably keyboard or mouse driver
		Irp->IoStatus.Status = STATUS_INVALID_DEVICE_REQUEST;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		return STATUS_INVALID_DEVICE_REQUEST;
	}
	// If this is a partition, get the physical disk.
	PUSBMS_PARTITION Partition = NULL;
	if (Ext->Signature == SIGNATURE_PARTITION) {
		Partition = (PUSBMS_PARTITION) Ext;
		Ext = Partition->PhysicalDisk;
	}
	
	// ensure this is actually a disk
	if (Ext->Signature != SIGNATURE_DISK) {
		Irp->IoStatus.Status = STATUS_INVALID_PARAMETER;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		return STATUS_INVALID_PARAMETER;
	}
	
	ULONG SectorSize = Ext->SectorSize;
	ULONG SectorMask = SectorSize - 1;	

	PIO_STACK_LOCATION Stack = IoGetCurrentIrpStackLocation(Irp);
	ULONG Length = Stack->Parameters.Read.Length;
	LARGE_INTEGER Offset = Stack->Parameters.Read.ByteOffset;
	ULONG SectorOffset = Offset.LowPart & SectorMask;
	
	// Ensure the read is aligned to sector size.
	if ((Length & SectorMask) != 0) {
		NTSTATUS Status = STATUS_INVALID_PARAMETER;
		if (Ext->DriveNotReady) Status = STATUS_DEVICE_NOT_READY;
		Irp->IoStatus.Status = Status;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		return Status;
	}
	
	// If this is a partition, fix up the offsets to be based on the disk.
	if (Partition != NULL) {
		// Get the ending offset of the read.
		LARGE_INTEGER OffsetEnd = Offset;
		OffsetEnd.QuadPart += Length;
		// Fix up the offset.
		Offset.QuadPart += Partition->StartingOffset.QuadPart;
		Stack->Parameters.Read.ByteOffset = Offset;
		// Get the partition length.
		LARGE_INTEGER PartitionLength = Partition->PartitionLength;
		// Ensure r/w in bounds of the partition.
		if (OffsetEnd.QuadPart >= PartitionLength.QuadPart) {
			NTSTATUS Status = STATUS_INVALID_PARAMETER;
			if (Ext->DriveNotReady) Status = STATUS_DEVICE_NOT_READY;
			Irp->IoStatus.Status = Status;
			IoCompleteRequest(Irp, IO_NO_INCREMENT);
			return Status;
		}
	}
	
	// Get the length of the disk.
	LARGE_INTEGER DiskLength = RtlEnlargedUnsignedMultiply(Ext->SectorSize, Ext->SectorCount);
	
	// Ensure offset + length is not beyond the end of the disk.
	LARGE_INTEGER OffsetEnd = Offset;
	OffsetEnd.QuadPart += Length;
	if (
		OffsetEnd.QuadPart > DiskLength.QuadPart
	) {
		NTSTATUS Status = STATUS_INVALID_PARAMETER;
		if (Ext->DriveNotReady) Status = STATUS_DEVICE_NOT_READY;
		Irp->IoStatus.Status = Status;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		return Status;
	}
	
	// Ensure sector number fits in 32 bits...
	LARGE_INTEGER SectorStart;
	SectorStart.QuadPart = (Stack->Parameters.Read.ByteOffset.QuadPart >> Ext->SectorShift);
	if (SectorStart.HighPart != 0) {
		Irp->IoStatus.Status = STATUS_INVALID_PARAMETER;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		return STATUS_INVALID_PARAMETER;
	}
	
	// All looks good, start the operation on the physical disk device.
	Irp->IoStatus.Information = 0;
	IoMarkIrpPending(Irp);
	IoStartPacket(Ext->DeviceObject, Irp, &SectorOffset, NULL);
	return STATUS_PENDING;
}

void MsIoWorkRoutine(PUSBMS_WORK_ITEM Parameter) {
	// Grab the parameters.
	PDEVICE_OBJECT DeviceObject = Parameter->DeviceObject;
	PIRP Irp = Parameter->Irp;
	PUSBMS_EXTENSION ExtDisk = Parameter->ExtDisk;
	PCONTROLLER_OBJECT CtrlObj = ExtDisk->Controller;
	PUSBMS_CONTROLLER Controller = CtrlObj->ControllerExtension;
	
	// Free the work item.
	ExFreePool(Parameter);
	
	// Run a test unit ready and request sense to make sure everything's fine.
	//PCONTROLLER_OBJECT CtrlObj = Ext->Controller;
	//PUSBMS_CONTROLLER Controller = CtrlObj->ControllerExtension;
	NTSTATUS Status = MspClearErrors(Controller, ExtDisk->Lun, 10);
	if (!NT_SUCCESS(Status)) {
		Irp->IoStatus.Status = Status;
		KeInsertQueueDpc(&ExtDisk->FinishDpc, Irp, CtrlObj);
		return;
	}
	
	// Perform the I/O operation on the work thread.
	PIO_STACK_LOCATION Stack = IoGetCurrentIrpStackLocation(Irp);
	
	// Calculate the sector offset and length.
	ULONG SectorShift = ExtDisk->SectorShift;
	ULONG SectorStart = (ULONG)(Stack->Parameters.Read.ByteOffset.QuadPart >> SectorShift);
	ULONG SectorCount = (ULONG)(Stack->Parameters.Read.Length >> SectorShift);
	//NTSTATUS Status = STATUS_INVALID_PARAMETER;
	if ((SectorCount << SectorShift) != Stack->Parameters.Read.Length) {
		Irp->IoStatus.Status = STATUS_INVALID_PARAMETER;
		KeInsertQueueDpc(&ExtDisk->FinishDpc, Irp, CtrlObj);
		return;
	}
	
	if (Stack->MajorFunction != IRP_MJ_DEVICE_CONTROL) {
		PVOID Buffer = MmGetSystemAddressForMdl(Irp->MdlAddress);
		if (Buffer == NULL) {
			Irp->IoStatus.Status = STATUS_INVALID_PARAMETER;
			Irp->IoStatus.Information = 0;
			KeInsertQueueDpc(&ExtDisk->FinishDpc, Irp, CtrlObj);
			return;
		}
		ULONG Transferred;
		ULONG Length = Stack->Parameters.Read.Length;
		if (Stack->MajorFunction == IRP_MJ_READ) {
			Status = MspLowRead(Controller, ExtDisk->Lun, SectorStart, SectorCount, SectorShift, Buffer);
		} else if (Stack->MajorFunction == IRP_MJ_WRITE) {
			Status = MspLowWrite(Controller, ExtDisk->Lun, SectorStart, SectorCount, SectorShift, Buffer);
		}
		Irp->IoStatus.Status = Status;
		if (NT_SUCCESS(Irp->IoStatus.Status)) {
			Irp->IoStatus.Information = Length;
		}
		KeInsertQueueDpc(&ExtDisk->FinishDpc, Irp, CtrlObj);
		return;
	}
	
	// This is verify.
	// Perform a READ(10) but throw the contents away, this should simulate verify.
	// VERIFY(10) does not allow specific LUN to be used, and some USB devices might not support it, so...
	Status = MspLowVerify(Controller, ExtDisk->Lun, SectorStart, SectorCount, SectorShift);
	Irp->IoStatus.Status = Status;
	KeInsertQueueDpc(&ExtDisk->FinishDpc, Irp, CtrlObj);
}

IO_ALLOCATION_ACTION MsIoSeizedController(PDEVICE_OBJECT DeviceObject, PIRP Irp, PVOID NullBase, PVOID Context) {
	PUSBMS_EXTENSION ExtDisk = (PUSBMS_EXTENSION) Context;
	
	// Touching disk needs to be done by a worker.
	// This is because we need to wait on events, and we are currently at DISPATCH_LEVEL.
	
	// Allocate the work item
	PUSBMS_WORK_ITEM WorkItem = (PUSBMS_WORK_ITEM)
		ExAllocatePool(NonPagedPool, sizeof(USBMS_WORK_ITEM));
	if (WorkItem == NULL) {
		// um.
		Irp->IoStatus.Status = STATUS_NO_MEMORY;
		// Complete the request.
		IoCompleteRequest(Irp, IO_DISK_INCREMENT);
		// Start next packet.
		IoStartNextPacket(DeviceObject, FALSE);
		// Release the controller on return so next packet can seize it.
		return DeallocateObject;
	}
	
	// Fill in the parameters.
	WorkItem->DeviceObject = DeviceObject;
	WorkItem->Irp = Irp;
	WorkItem->ExtDisk = ExtDisk;
	
	// Initialise the ExWorkItem
	ExInitializeWorkItem(
		&WorkItem->WorkItem,
		(PWORKER_THREAD_ROUTINE) MsIoWorkRoutine,
		WorkItem
	);
	
	// Queue it, and keep the controller object held.
	BOOLEAN IsForPaging = (Irp->Flags & IRP_PAGING_IO) != 0;
	ExQueueWorkItem(&WorkItem->WorkItem, IsForPaging ? CriticalWorkQueue : DelayedWorkQueue);
	return KeepObject;
}

void MsStartIo(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
	PUSBMS_EXTENSION Ext = (PUSBMS_EXTENSION)DeviceObject->DeviceExtension;
	// Ensure the device extension is actually valid.
	if (Ext == NULL || (
		Ext->Signature != SIGNATURE_PARTITION &&
		Ext->Signature != SIGNATURE_DISK
	) || Ext->Controller == NULL) {
		// this is probably keyboard or mouse driver
		Irp->IoStatus.Status = STATUS_INVALID_DEVICE_REQUEST;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		IoStartNextPacket(DeviceObject, FALSE);
		return;
	}
	// If this is a partition, get the physical disk.
	PUSBMS_PARTITION Partition = NULL;
	if (Ext->Signature == SIGNATURE_PARTITION) {
		Partition = (PUSBMS_PARTITION) Ext;
		Ext = Partition->PhysicalDisk;
	}
	
	// ensure this is actually a disk
	if (Ext->Signature != SIGNATURE_DISK) {
		Irp->IoStatus.Status = STATUS_INVALID_PARAMETER;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		IoStartNextPacket(DeviceObject, FALSE);
		return;
	}
	
	// Seize the controller object to obtain exclusive access to its DDR buffer.
	IoAllocateController(
		Ext->Controller,
		DeviceObject,
		(PDRIVER_CONTROL)MsIoSeizedController,
		Ext
	);
}

void MspUpdateDevices(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
	PUSBMS_EXTENSION Ext = (PUSBMS_EXTENSION)DeviceObject->DeviceExtension;
	PUSBMS_PARTITION Partition = NULL;
	if (Ext->Signature == SIGNATURE_PARTITION) {
		Partition = (PUSBMS_PARTITION)DeviceObject->DeviceExtension;
		Ext = Partition->PhysicalDisk;
	}
	if (Ext->Signature != SIGNATURE_DISK) {
		return;
	}

	// Ensure that this device object describes a hard disk
	if (DeviceObject->DeviceType == FILE_DEVICE_CD_ROM) return;
	if ((DeviceObject->Characteristics & FILE_FLOPPY_DISKETTE) != 0) return;
	
	PDRIVE_LAYOUT_INFORMATION PartitionList = (PDRIVE_LAYOUT_INFORMATION) Irp->AssociatedIrp.SystemBuffer;
	
	ULONG PartitionCount = ((PartitionList->PartitionCount + 3) / 4) * 4;
	
	// Zero all the partition numbers.
	for (ULONG Partition = 0; Partition < PartitionCount; Partition++) {
		PartitionList->PartitionEntry[Partition].PartitionNumber = 0;
	}
	
	// Walk through the partitions to determine if any existing partitions were changed or deleted.
	ULONG LastPartitionNumber = 0;
	PUSBMS_PARTITION LastPartition = NULL;
	for (PUSBMS_PARTITION This = Ext->NextPartition; This != NULL; This = This->NextPartition) {
		LastPartition = This;
		
		if (This->PartitionNumber > LastPartitionNumber)
			LastPartitionNumber = This->PartitionNumber;
		
		// If the partition length is zero then it's unused.
		if (This->PartitionLength.QuadPart == 0) continue;
		
		// Look for a match in the partition list.
		BOOLEAN Found = FALSE;
		PPARTITION_INFORMATION PartitionEntry = NULL;
		for (ULONG Partition = 0; Partition < PartitionCount; Partition++) {
			PartitionEntry = &PartitionList->PartitionEntry[Partition];
			
			// Skip this if it's empty.
			if (PartitionEntry->PartitionType == PARTITION_ENTRY_UNUSED) continue;
			
			// Skip this if it's an extended partition.
			if (IsContainerPartition(PartitionEntry->PartitionType)) continue;
			
			// Offset is equal?
			if (PartitionEntry->StartingOffset.QuadPart != This->StartingOffset.QuadPart) continue;
			
			// Length is equal?
			if (PartitionEntry->PartitionLength.QuadPart != This->PartitionLength.QuadPart) continue;
			
			// Found the correct partition.
			Found = TRUE;
			PartitionEntry->PartitionNumber = This->PartitionNumber;
			break;
		}
		
		if (Found) {
			// Change the partition type if needed.
			if (PartitionEntry->RewritePartition)
				This->PartitionType = PartitionEntry->PartitionType;
		} else {
			// Didn't find a match, thus this partition was deleted.
			This->PartitionLength.QuadPart = 0;
		}
	}
	
	// Walk through the partitions to determine if any new partitions were created.
	for (ULONG Partition = 0; Partition < PartitionCount; Partition++) {
		PPARTITION_INFORMATION PartitionEntry = &PartitionList->PartitionEntry[Partition];
		
		// Skip this if it's empty.
		if (PartitionEntry->PartitionType == PARTITION_ENTRY_UNUSED) continue;
		
		// Skip this if it's an extended partition.
		if (IsContainerPartition(PartitionEntry->PartitionType)) continue;
		
		// Skip this if we know about it already
		if (PartitionEntry->PartitionNumber != 0) continue;
		
		// Find an existing, unused, partition.
		ULONG PartitionNumber = 0;
		PUSBMS_PARTITION ModifiedPartition = NULL;
		for (PUSBMS_PARTITION This = Ext->NextPartition; This != NULL; This = This->NextPartition) {
			if (This->PartitionLength.QuadPart != 0) continue;
			PartitionNumber = This->PartitionNumber;
			ModifiedPartition = This;
			break;
		}
		
		// If none was found then one needs to be created.
		if (PartitionNumber == 0) {
			LastPartitionNumber++;
			PartitionNumber = LastPartitionNumber;
			
			// Initialise the pathname.
			CCHAR NtName[MAXIMUM_FILENAME_LENGTH];
			sprintf(NtName, "\\Device\\Harddisk%d\\Partition%d", Ext->DiskNumber, PartitionNumber);
			STRING NtNameStr;
			RtlInitString(&NtNameStr, NtName);
			UNICODE_STRING NtNameUs;
			NTSTATUS Status = RtlAnsiStringToUnicodeString(&NtNameUs, &NtNameStr, TRUE);
			if (!NT_SUCCESS(Status)) continue;
			
			// Create the device object.
			PDEVICE_OBJECT NewDevice;
			Status = IoCreateDevice(DeviceObject->DriverObject, sizeof(USBMS_PARTITION), &NtNameUs, FILE_DEVICE_DISK, 0, FALSE, &NewDevice);
			RtlFreeUnicodeString(&NtNameUs);
			
			if (!NT_SUCCESS(Status)) continue;
			
			// Initialise the device object.
			NewDevice->Flags |= DO_DIRECT_IO;
			NewDevice->StackSize = DeviceObject->StackSize;
			NewDevice->Flags &= ~DO_DEVICE_INITIALIZING;
			
			// Initialise the device extension.
			PUSBMS_PARTITION NewPartition = (PUSBMS_PARTITION) NewDevice->DeviceExtension;
			NewPartition->Signature = SIGNATURE_PARTITION;
			NewPartition->PhysicalDisk = Ext;
			if (LastPartition != NULL) LastPartition->NextPartition = NewPartition;
			else Ext->NextPartition = NewPartition;
			NewPartition->NextPartition = NULL;
			
			ModifiedPartition = NewPartition;
			LastPartition = NewPartition;
		}
		// Update the partition information.
		ModifiedPartition->PartitionNumber = PartitionNumber;
		ModifiedPartition->PartitionType = PartitionEntry->PartitionType;
		ModifiedPartition->StartingOffset = PartitionEntry->StartingOffset;
		ModifiedPartition->PartitionLength = PartitionEntry->PartitionLength;
		ModifiedPartition->HiddenSectors = PartitionEntry->HiddenSectors;
		
		PartitionEntry->PartitionNumber = PartitionNumber;
	}
}

NTSTATUS MsDiskDeviceControl(PDEVICE_OBJECT DeviceObject, PIRP Irp) {
	PIO_STACK_LOCATION Stack = IoGetCurrentIrpStackLocation(Irp);
	PUSBMS_EXTENSION Ext = (PUSBMS_EXTENSION)DeviceObject->DeviceExtension;
	PUSBMS_PARTITION Partition = NULL;
	// Ensure the device extension is actually valid.
	if (Ext == NULL || (
		Ext->Signature != SIGNATURE_PARTITION &&
		Ext->Signature != SIGNATURE_DISK
	) || Ext->Controller == NULL) {
		// this is probably keyboard or mouse driver
		Irp->IoStatus.Status = STATUS_INVALID_DEVICE_REQUEST;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		return STATUS_INVALID_DEVICE_REQUEST;
	}
	if (Ext->Signature == SIGNATURE_PARTITION) {
		Partition = (PUSBMS_PARTITION)DeviceObject->DeviceExtension;
		Ext = Partition->PhysicalDisk;
	}
	if (Ext->Signature != SIGNATURE_DISK) {
		Irp->IoStatus.Status = STATUS_INVALID_PARAMETER;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		return STATUS_INVALID_PARAMETER;
	}
	PCONTROLLER_OBJECT CtrlObj = Ext->Controller;
	PUSBMS_CONTROLLER Controller = (PUSBMS_CONTROLLER)CtrlObj->ControllerExtension;
	
	NTSTATUS Status = STATUS_SUCCESS;
	ULONG LenIn = Stack->Parameters.DeviceIoControl.InputBufferLength;
	ULONG LenOut = Stack->Parameters.DeviceIoControl.OutputBufferLength;
	BOOLEAN IsCdRom = DeviceObject->DeviceType == FILE_DEVICE_CD_ROM;
	BOOLEAN IsFloppy = (DeviceObject->Characteristics & FILE_FLOPPY_DISKETTE) != 0;
	BOOLEAN IsRemovable = DeviceObject->Characteristics & FILE_REMOVABLE_MEDIA;
	BOOLEAN FallThrough = FALSE;
	
	if (IsCdRom) {
		switch (Stack->Parameters.DeviceIoControl.IoControlCode) {
		case IOCTL_CDROM_GET_DRIVE_GEOMETRY:
		case IOCTL_CDROM_CHECK_VERIFY:
			FallThrough = TRUE;
			break;
		// Multisession / CD audio stuff : unimplemented.
		// We only support a single data track as .ISO.
		// If this stuff is unimplemented cdfs.sys does fall back to such anyways.
		// case IOCTL_CDROM_GET_LAST_SESSION:
		// case IOCTL_CDROM_READ_TOC:
		// case IOCTL_CDROM_PLAY_AUDIO_MSF:
		// case IOCTL_CDROM_SEEK_AUDIO_MSF:
		// case IOCTL_CDROM_PAUSE_AUDIO:
		// case IOCTL_CDROM_RESUME_AUDIO:
		// case IOCTL_CDROM_READ_Q_CHANNEL:
		// case IOCTL_CDROM_GET_CONTROL:
		// case IOCTL_CDROM_GET_VOLUME:
		// case IOCTL_CDROM_SET_VOLUME:
		// case IOCTL_CDROM_STOP_AUDIO:

		default:
			Status = STATUS_INVALID_DEVICE_REQUEST;
			break;
		}
	}
	else if (IsFloppy) {
		// we only have to implement this because things want to read from a floppy drive...
		switch (Stack->Parameters.DeviceIoControl.IoControlCode) {
		case IOCTL_DISK_FORMAT_TRACKS:
		case IOCTL_DISK_FORMAT_TRACKS_EX:
        	case IOCTL_DISK_IS_WRITABLE:
			Status = STATUS_MEDIA_WRITE_PROTECTED;
			break;
		case IOCTL_DISK_CHECK_VERIFY:
		case IOCTL_DISK_GET_DRIVE_GEOMETRY:
		case IOCTL_DISK_GET_MEDIA_TYPES:
			FallThrough = TRUE;
			break;
		default:
			Status = STATUS_INVALID_DEVICE_REQUEST;
			break;
		}
	}
	else {
		FallThrough = TRUE;
	}
	
	if (!FallThrough) {
		Irp->IoStatus.Status = Status;
		IoCompleteRequest(Irp, IO_NO_INCREMENT);
		return Status;
	}
	
	// not cdrom or floppy, or we fell through to common code
	switch (Stack->Parameters.DeviceIoControl.IoControlCode) {
	case IOCTL_DISK_IS_WRITABLE:
		// Run check verify if removable media to update the WP bit.
		if (IsRemovable) Status = MspCheckVerify(DeviceObject, Controller, Ext);
		if (Ext->WriteProtected) Status = STATUS_MEDIA_WRITE_PROTECTED;
		break;
	case IOCTL_DISK_GET_MEDIA_TYPES:
		if (!IsFloppy) {
			Status = STATUS_INVALID_DEVICE_REQUEST;
			break;
		}
		// fall through, we only support the one geometry, which is the fatfs image we say is a floppy
	case IOCTL_DISK_GET_DRIVE_GEOMETRY:
	case IOCTL_CDROM_GET_DRIVE_GEOMETRY:
		{
			if (LenOut < sizeof(DISK_GEOMETRY)) {
				Status = STATUS_INVALID_PARAMETER;
				break;
			}
			// Get the current information if needded.
			if (IsRemovable) {
				Status = MspCheckVerify(DeviceObject, Controller, Ext);
				if (!NT_SUCCESS(Status)) break;
			}
			PDISK_GEOMETRY Geometry = (PDISK_GEOMETRY) Irp->AssociatedIrp.SystemBuffer;
			// Set up the disk geometry.
			if (DeviceObject->Characteristics & FILE_REMOVABLE_MEDIA) {
				Geometry->MediaType = RemovableMedia;
			} else {
				Geometry->MediaType = FixedMedia;
			}
			// Default sectors per track / tracks per head as per what SCSI devices do
			Geometry->SectorsPerTrack = 32;
			Geometry->TracksPerCylinder = 64;
			// set BytesPerSector to the specified value.
			// For an ISO use 2048 byte sectors as expected; it's a multiple of 512 anyway
			// We will allow rws aligned to 512 no matter what.
			LARGE_INTEGER DeviceLength;
			if (Partition != NULL) DeviceLength = Partition->PartitionLength;
			else {
				DeviceLength.QuadPart = Ext->SectorCount;
				DeviceLength.QuadPart <<= Ext->SectorShift;
			}
			Geometry->BytesPerSector = Ext->SectorSize;
			Geometry->Cylinders.LowPart = Ext->SectorCount / (32 * 64);
			Geometry->Cylinders.HighPart = 0;
			// all done.
			Irp->IoStatus.Information = sizeof(*Geometry);
		}
		break;
	case IOCTL_DISK_CHECK_VERIFY:
	case IOCTL_CDROM_CHECK_VERIFY:
		{
			// If device is not removable, just return success.
			if (!IsRemovable) break;
			// Run check verify.
			Status = MspCheckVerify(DeviceObject, Controller, Ext);
		}
		break;
	case IOCTL_DISK_VERIFY:
		{
			if (LenIn < sizeof(VERIFY_INFORMATION)) {
				Status = STATUS_INVALID_PARAMETER;
				break;
			}
			Irp->IoStatus.Information = 0;
			IoMarkIrpPending(Irp);
			// Make it look like a rw request for the async code.
			PVERIFY_INFORMATION VerifyInfo = (PVERIFY_INFORMATION)Irp->AssociatedIrp.SystemBuffer;
			Stack->Parameters.Read.Length = VerifyInfo->Length;
			Stack->Parameters.Read.ByteOffset.QuadPart = VerifyInfo->StartingOffset.QuadPart;
			if (Partition != NULL)
				Stack->Parameters.Read.ByteOffset.QuadPart += Partition->StartingOffset.QuadPart;
			// make sure the end looks ok
			LARGE_INTEGER DeviceLength;
			DeviceLength.QuadPart = Ext->SectorSize;
			DeviceLength.QuadPart <<= Ext->SectorShift;
			LARGE_INTEGER OffsetEnd = Stack->Parameters.Read.ByteOffset;
			OffsetEnd.QuadPart += Stack->Parameters.Read.Length;
			if (OffsetEnd.QuadPart > DeviceLength.QuadPart || (Stack->Parameters.Read.Length & (Ext->SectorSize - 1) != 0)) {
				Status = STATUS_INVALID_PARAMETER;
				break;
			}
			ULONG SectorOffset = Stack->Parameters.Read.ByteOffset.QuadPart & (Ext->SectorSize - 1);
			IoStartPacket(Ext->DeviceObject, Irp, &SectorOffset, NULL);
			return STATUS_PENDING;
		}
		break;
	case IOCTL_DISK_GET_PARTITION_INFO:
		{
			if (LenOut < sizeof(PARTITION_INFORMATION)) {
				Status = STATUS_INVALID_PARAMETER;
				break;
			}
			if (Partition == NULL) {
				Status = STATUS_INVALID_DEVICE_REQUEST;
				break;
			}
			PPARTITION_INFORMATION Info = (PPARTITION_INFORMATION)Irp->AssociatedIrp.SystemBuffer;
			Info->PartitionType = Partition->PartitionType;
			Info->StartingOffset = Partition->StartingOffset;
			Info->PartitionLength = Partition->PartitionLength;
			Info->HiddenSectors = Partition->HiddenSectors;
			Info->PartitionNumber = Partition->PartitionNumber;
			
			Irp->IoStatus.Information = sizeof(*Info);
		}
		break;
	case IOCTL_DISK_SET_PARTITION_INFO:
		{
			if (LenIn < sizeof(SET_PARTITION_INFORMATION)) {
				Status = STATUS_INVALID_PARAMETER;
				break;
			}
			if (Partition == NULL) {
				Status = STATUS_INVALID_DEVICE_REQUEST;
				break;
			}
			PSET_PARTITION_INFORMATION Info = (PSET_PARTITION_INFORMATION)Irp->AssociatedIrp.SystemBuffer;
			Status = IoSetPartitionInformation(Ext->DeviceObject, Ext->SectorSize, (ULONG) Partition->PartitionNumber, Info->PartitionType);
			if (NT_SUCCESS(Status)) Partition->PartitionType = Info->PartitionType;
		}
		break;
	case IOCTL_DISK_GET_DRIVE_LAYOUT:
		{
			if (LenOut < sizeof(DRIVE_LAYOUT_INFORMATION)) {
				Status = STATUS_INVALID_PARAMETER;
				break;
			}
			
			if (IsFloppy) {
				Status = STATUS_INVALID_DEVICE_REQUEST;
				break;
			}
			
			PDRIVE_LAYOUT_INFORMATION PartitionList;
			Status = IoReadPartitionTable(Ext->DeviceObject, Ext->SectorSize, FALSE, &PartitionList);
			if (!NT_SUCCESS(Status)) break;
			
			// what's the size of the buffer we were given?
			ULONG Size = __builtin_offsetof(DRIVE_LAYOUT_INFORMATION, PartitionEntry);
			Size += PartitionList->PartitionCount * sizeof(PartitionList->PartitionEntry[0]);
			if (LenOut < Size) {
				Status = STATUS_BUFFER_TOO_SMALL;
				break;
			}
			Status = STATUS_SUCCESS;
			RtlMoveMemory(Irp->AssociatedIrp.SystemBuffer, PartitionList, Size);
			Irp->IoStatus.Information = Size;
			MspUpdateDevices(DeviceObject, Irp);
			ExFreePool(PartitionList);
		}
		break;
	case IOCTL_DISK_SET_DRIVE_LAYOUT:
		{
			if (LenIn < sizeof(DRIVE_LAYOUT_INFORMATION)) {
				Status = STATUS_INVALID_PARAMETER;
				break;
			}
			if (IsFloppy) {
				Status = STATUS_INVALID_DEVICE_REQUEST;
				break;
			}
			
			PDRIVE_LAYOUT_INFORMATION PartitionList = (PDRIVE_LAYOUT_INFORMATION) Irp->AssociatedIrp.SystemBuffer;
			
			MspUpdateDevices(DeviceObject, Irp);
			
			Status = IoWritePartitionTable(Ext->DeviceObject, Ext->SectorSize, 32, 64, PartitionList);
			if (!NT_SUCCESS(Status)) break;
			Irp->IoStatus.Information = Stack->Parameters.DeviceIoControl.OutputBufferLength;			
		}
		break;
	case IOCTL_DISK_INTERNAL_SET_VERIFY:
		if (Irp->RequestorMode != KernelMode) break;
		DeviceObject->Flags |= DO_VERIFY_VOLUME;
		break;
	case IOCTL_DISK_INTERNAL_CLEAR_VERIFY:
		if (Irp->RequestorMode != KernelMode) break;
		DeviceObject->Flags &= ~DO_VERIFY_VOLUME;
		break;
	case IOCTL_SCSI_GET_ADDRESS:
	{
		if (LenOut < sizeof(SCSI_ADDRESS)) {
			Status = STATUS_INVALID_PARAMETER;
			break;
		}
		
		PSCSI_ADDRESS ScsiAddress = (PSCSI_ADDRESS) Irp->AssociatedIrp.SystemBuffer;
		
		ScsiAddress->Length = sizeof(ScsiAddress);
		ScsiAddress->PortNumber = 0;
		if (IsFloppy) ScsiAddress->PathId = 0;
		else if (IsCdRom) ScsiAddress->PathId = 1;
		else ScsiAddress->PathId = 2;
		ScsiAddress->TargetId = Ext->DiskNumber;
		ScsiAddress->Lun = Ext->Lun;
		
		break;
	}
	default:
		Status = STATUS_INVALID_DEVICE_REQUEST;
		break;
	}
	
	Irp->IoStatus.Status = Status;
	IoCompleteRequest(Irp, IO_NO_INCREMENT);
	return Status;
}

NTSTATUS MspInitDevice(
	PDRIVER_OBJECT DriverObject,
	IOS_USB_HANDLE DeviceHandle,
	ULONG ArcKey,
	PVOID UsbBuffer,
	PULONG UsbOffset,
	PCONFIGURATION_INFORMATION IoConfig
) {
	// Open device.
	NTSTATUS Status = UlOpenDevice(DeviceHandle);
	if (!NT_SUCCESS(Status)) return Status;
	
	do {
		// Get device descriptors.
		USB_DEVICE_DESC Descriptors;
		Status = UlGetDescriptors(DeviceHandle, &Descriptors);
		if (!NT_SUCCESS(Status)) break;
		
		if (Descriptors.Device.bNumConfigurations == 0) {
			Status = STATUS_NO_SUCH_DEVICE;
			break;
		}
		
		PUSB_CONFIGURATION Config = &Descriptors.Config;
		if (Config->bNumInterfaces == 0) {
			Status = STATUS_NO_SUCH_DEVICE;
			break;
		}
		PUSB_INTERFACE Interface = &Descriptors.Interface;
		if (Interface->bInterfaceClass != USB_CLASS_MASS_STORAGE) {
			Status = STATUS_NO_SUCH_DEVICE;
			break;
		}
		if (Interface->bInterfaceProtocol != MASS_STORAGE_BULK_ONLY) {
			Status = STATUS_NO_SUCH_DEVICE;
			break;
		}
		if (Interface->bNumEndpoints < 2) {
			Status = STATUS_NO_SUCH_DEVICE;
			break;
		}
		
		BOOLEAN IsUsb2 = FALSE;
		UCHAR EndpointIn = 0, EndpointOut = 0;
		for (ULONG i = 0; i < Interface->bNumEndpoints; i++) {
			PUSB_ENDPOINT Endpoint = &Descriptors.Endpoints[i];
			if (Endpoint->bmAttributes != USB_ENDPOINT_BULK) continue;
			if ((Endpoint->bEndpointAddress & USB_ENDPOINT_IN) != 0) {
				EndpointIn = Endpoint->bEndpointAddress;
			} else {
				EndpointOut = Endpoint->bEndpointAddress;
				IsUsb2 = Endpoint->wMaxPacketSize > 64;
			}
		}
		
		if (EndpointIn == 0 || EndpointOut == 0) {
			Status = STATUS_NO_SUCH_DEVICE;
			break;
		}
		
		// Ignore failing status as some devices do not support these
		UCHAR CurrConfig = 0;
		UlGetConfiguration(DeviceHandle, &CurrConfig);
		if (CurrConfig != Config->bConfigurationValue) {
			UlSetConfiguration(DeviceHandle, Config->bConfigurationValue);
		}
		if (Interface->bAlternateSetting != 0) {
			UlSetAlternativeInterface(DeviceHandle, Interface->bInterfaceNumber, Interface->bAlternateSetting);
		}
		// libogc implementation sends a reset here for usb1;
		// however we're only using usbv5, so is that actually needed?
		// let's do it anyway
		// update: do it no matter what, we're probably coming from an arc firmware that used it.
		MspReset(DeviceHandle, Interface->bInterfaceNumber, EndpointIn, EndpointOut);
		
		PUCHAR pMaxLun = (PUCHAR)HalIopAlloc(sizeof(UCHAR));
		if (pMaxLun == NULL) {
			Status = STATUS_INSUFFICIENT_RESOURCES;
			break;
		}
		Status = MspCtrlMsgTimeout(DeviceHandle,
			(USB_CTRLTYPE_DIR_DEVICE2HOST | USB_CTRLTYPE_TYPE_CLASS | USB_CTRLTYPE_REC_INTERFACE),
			USBSTORAGE_GET_MAX_LUN,
			0,
			Interface->bInterfaceNumber,
			1,
			pMaxLun,
			USBSTORAGE_TIMEOUT
		);
		ULONG MaxLun = *pMaxLun + 1;
		HalIopFree(pMaxLun);
		if (Status == STATUS_IO_TIMEOUT) {
			Status = STATUS_NO_SUCH_DEVICE;
			break;
		}
		if (!NT_SUCCESS(Status)) MaxLun = 1;
		
		// This device is working USB mass storage device.
		// We can now do SCSI inquiry and read capacity on all LUNs;
		// and create disk device for all LUNs.
		
		// First, create the controller for this device.
		PCONTROLLER_OBJECT Controller = IoCreateController(sizeof(USBMS_CONTROLLER));
		if (Controller == NULL) {
			Status = STATUS_INSUFFICIENT_RESOURCES;
			break;
		}
		// Grab the extension.
		PUSBMS_CONTROLLER Extension = (PUSBMS_CONTROLLER)
			Controller->ControllerExtension;
		
		// Fill in the parameters.
		Extension->DeviceHandle = DeviceHandle;
		Extension->EndpointIn = EndpointIn;
		Extension->EndpointOut = EndpointOut;
		Extension->Interface = Interface->bInterfaceNumber;
		Extension->MaxSize = IsUsb2 ? MAX_TRANSFER_SIZE_V2 : MAX_TRANSFER_SIZE_V1;
		Extension->Buffer = (PVOID) (*UsbOffset + (ULONG)UsbBuffer);
		*UsbOffset += Extension->MaxSize;
		
		BOOLEAN HasValidLun = FALSE;
		BOOLEAN EntryPointsSet = DriverObject->DriverStartIo == MsStartIo;
		for (int lun = 0; lun < MaxLun; lun++) {
			USBMS_DISK_TYPE DiskType = USBMS_DISK_UNKNOWN;
			Status = MspGetDiskType(Extension, lun, &DiskType, USBSTORAGE_TIMEOUT);
			if (!NT_SUCCESS(Status)) continue;

			// Get the drive capacity, if needed.
			ULONG SectorSize = 0;
			ULONG SectorCount = 0;
			Status = MspReadCapacity(Extension, lun, &SectorSize, &SectorCount, USBSTORAGE_TIMEOUT);
			if (!NT_SUCCESS(Status) && DiskType == USBMS_DISK_OTHER_FIXED) continue;
			
			// Set up the usbms dispatch callbacks.
			if (!EntryPointsSet) {
				DriverObject->DriverStartIo = MsStartIo;
				DriverObject->MajorFunction[IRP_MJ_READ] = MsDiskRw;
				DriverObject->MajorFunction[IRP_MJ_WRITE] = MsDiskRw;
				DriverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] = MsDiskDeviceControl;
				EntryPointsSet = TRUE;
			}
			// Create the device.
			Status = MspDiskCreate(DriverObject, Controller, lun, SectorSize, SectorCount, IoConfig, ArcKey, DiskType);
			if (!NT_SUCCESS(Status)) continue;
			
			HasValidLun = TRUE;
		}
		if (HasValidLun) Status = STATUS_SUCCESS;
	} while (FALSE);
	//if (!NT_SUCCESS(Status)) UlCloseDevice(DeviceHandle);
	return Status;
}

NTSTATUS UlmsInit(PDRIVER_OBJECT DriverObject) {
	// Map USB buffer uncached.
	// (We will always be using 32-bit writes to it.)
	// (...as sector size is always 32-bit aligned)
	PHYSICAL_ADDRESS UsbBufferPhysical = {.QuadPart = 0};
	UsbBufferPhysical.LowPart = USB_BUFFER_PHYS_START;
	PVOID UsbBuffer = MmMapIoSpace( UsbBufferPhysical, USB_BUFFER_LENGTH, MmNonCached );
	if (UsbBuffer == NULL) {
		// well,  we can't do anything!
		return STATUS_NO_SUCH_DEVICE;
	}
	// Get device list.
	
	PIOS_USB_DEVICE_ENTRY Entries = (PIOS_USB_DEVICE_ENTRY)
		HalIopAlloc(sizeof(IOS_USB_DEVICE_ENTRY_MAX));
	if (Entries == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	RtlZeroMemory(Entries, sizeof(IOS_USB_DEVICE_ENTRY_MAX));
	
	UCHAR NumVen = 0;
	UlGetDeviceList(Entries, USB_COUNT_DEVICES, USB_CLASS_MASS_STORAGE, &NumVen);
	
	
	// Get the configuration information.
	PCONFIGURATION_INFORMATION Config = IoGetConfigurationInformation();
	
	ULONG UsbOffset = 0;
	UCHAR NumMs = 0;
	for (ULONG i = 0; i < NumVen; i++) {
		USHORT Vid = Entries[i].VendorId;
		USHORT Pid = Entries[i].ProductId;
		if (Vid == 0 || Pid == 0) continue;
		if (Vid == 0x0b95 && Pid == 0x7720) {
			// this is actually a USB network adapter
			continue;
		}
		
		// Create the ARC key, which is a 32-bit value:
		// top 16 bits is vendor ID and lower 16 bits is product ID
		ULONG ArcKey = Vid;
		ArcKey <<= 16;
		ArcKey |= Pid;
		// Check if this is a USB mass storage device and create the disk device object if so.
		NTSTATUS Status = MspInitDevice(DriverObject, Entries[i].DeviceHandle, ArcKey, UsbBuffer, &UsbOffset, Config);
		if (NT_SUCCESS(Status)) {
			// successful, device was created.
			NumMs++;
		}
	}
	
	// If no mass storage devices were found, then return as much.
	if (NumMs == 0) return STATUS_NO_SUCH_DEVICE;
	
	return STATUS_SUCCESS;
}