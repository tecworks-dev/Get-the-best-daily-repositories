// RVL/Cafe USB Mass Storage by USBv5
#include <stdio.h>
#include <memory.h>
#include "arc.h"
#include "types.h"
#include "runtime.h"
#include "pxi.h"
#include "timer.h"
#include "ios_usb.h"
#include "ios_usb_ms.h"

enum {
	CBW_SIZE = 31,
	CBW_SIGNATURE = 0x43425355,
	CBW_IN = (1 << 7),
	CBW_OUT = (0 << 7)
};

enum {
	CSW_SIZE = 13,
	CSW_SIGNATURE = 0x53425355,
};

enum {
	SCSI_TEST_UNIT_READY = 0x00,
	SCSI_REQUEST_SENSE = 0x03,
	SCSI_INQUIRY = 0x12,
	SCSI_MODE_SENSE = 0x1A,
	SCSI_START_STOP = 0x1B,
	SCSI_READ_FORMAT_CAPACITY = 0x23,
	SCSI_READ_CAPACITY = 0x25,
	SCSI_READ_10 = 0x28,
	SCSI_WRITE_10 = 0x2A
};

enum {
	SCSI_SENSE_REPLY_SIZE = 18,
	SCSI_SENSE_NOT_READY = 0x02,
	SCSI_SENSE_MEDIUM_ERROR = 0x03,
	SCSI_SENSE_HARDWARE_ERROR = 0x04,
	SCSI_SENSE_UNIT_ATTENTION = 0x06,
};

enum {
	USB_CLASS_MASS_STORAGE = 0x08,
	MASS_STORAGE_RBC_COMMANDS = 0x01,
	MASS_STORAGE_ATA_COMMANDS,
	MASS_STORAGE_QIC_COMMANDS,
	MASS_STORAGE_UFI_COMMANDS,
	MASS_STORAGE_SFF8070_COMMANDS,
	MASS_STORAGE_SCSI_COMMANDS,
	MASS_STORAGE_BULK_ONLY = 0x50
};

enum {
	USBSTORAGE_GET_MAX_LUN = 0xFE,
	USBSTORAGE_RESET = 0xFF,
};

enum {
	USB_ENDPOINT_BULK = 0x02,
};

enum {
	USBSTORAGE_CYCLE_RETRIES = 3,
	USBSTORAGE_TIMEOUT = 2
};

enum {
	MAX_TRANSFER_SIZE_V2 = 0x4000, // 4 pages
	MAX_TRANSFER_SIZE_V1 = 0x1000, // 1 page
};

enum {
	// 1MB buffer at start of DDR;
	// however due to IOS USBv5 bug the first 32 bytes cannot be used.
	// Thus, ignore the first page.
	// Anyway, MAX_TRANSFER_SIZE_V2 * USB_COUNT_DEVICES == 0x80000,
	// so even with the first page unused,
	// buffers for every possible device are guaranteed to fit.
	// Start half way through this buffer, let SDMC driver use the first half.
	USB_BUFFER_PHYS_START = 0x10080000, // 0x10001000,
	USB_BUFFER_LENGTH = 0x80000, // 0x100000 - 0x1000,
	USB_BUFFER_PHYS_END = USB_BUFFER_PHYS_START + USB_BUFFER_LENGTH
};

typedef struct ARC_LE _USBMS_CBW {
	ULONG Signature;
	ULONG Tag;
	ULONG Length;
	UCHAR Flags;
	UCHAR Lun;
	UCHAR CbLength;
	UCHAR Cb[16];
} USBMS_CBW, * PUSBMS_CBW;
_Static_assert(sizeof(USBMS_CBW) == 32);

typedef struct ARC_LE _USBMS_CSW {
	ULONG Signature;
	ULONG Tag;
	ULONG DataResidue;
	UCHAR Status;
} USBMS_CSW, * PUSBMS_CSW;
_Static_assert(sizeof(USBMS_CSW) == 16);

typedef struct ARC_BE ARC_PACKED _SCSI_FORMAT_CAPACITY_ENTRY {
	ULONG NumberOfBlocks;
	UCHAR Flags;
	UCHAR BlockLengthHigh;
	USHORT BlockLengthLow;
} SCSI_FORMAT_CAPACITY_ENTRY, * PSCSI_FORMAT_CAPACITY_ENTRY;

typedef struct ARC_BE ARC_PACKED _SCSI_FORMAT_CAPACITY_LIST {
	UCHAR Reserved[3];
	UCHAR Length;
	SCSI_FORMAT_CAPACITY_ENTRY Entries[0];
} SCSI_FORMAT_CAPACITY_LIST, * PSCSI_FORMAT_CAPACITY_LIST;

typedef struct ARC_BE _SCSI_READ_CAPACITY_RES {
	ULONG SectorCount;
	ULONG SectorSize;
} SCSI_READ_CAPACITY_RES, * PSCSI_READ_CAPACITY_RES;

typedef enum {
	USBMS_DISK_UNKNOWN,
	USBMS_DISK_FLOPPY,
	USBMS_DISK_CDROM,
	USBMS_DISK_OTHER_FIXED,
	USBMS_DISK_OTHER_REMOVABLE
} USBMS_DISK_TYPE, *PUSBMS_DISK_TYPE;

typedef struct _USBMS_LUN {
	USBMS_DISK_TYPE DiskType;
	ULONG SectorSize;
	ULONG SectorShift;
	ULONG SectorCount;
	UCHAR RealLun;
	bool WriteProtected;
	bool DriveNotReady;
	bool LunValid;
} USBMS_LUN, *PUSBMS_LUN;

enum {
	USBMS_MAX_LUNS = 8
};

struct _USBMS_CONTROLLER {
	IOS_USB_HANDLE DeviceHandle;
	PVOID Buffer;
	ULONG MaxSize;
	ULONG ArcKey;
	ULONG NumberOfLuns;
	UCHAR EndpointIn;
	UCHAR EndpointOut;
	UCHAR Interface;
	UCHAR CbwTag;
	USBMS_LUN Luns[USBMS_MAX_LUNS];
};

static USBMS_CONTROLLER s_MassStorageDevices[USB_COUNT_DEVICES];

static void ZeroMemory32(void* buffer, ULONG length) {
	if ((length & 3) != 0) {
		memset(buffer, 0, length);
		return;
	}
	length /= sizeof(ULONG);
	PULONG buf32 = (PULONG)buffer;
	for (ULONG i = 0; i < length; i++) buf32[i] = 0;
}

static void memcpy32(void* dest, const void* src, ULONG len) {
	PULONG dest32 = (PULONG)dest;
	const ULONG* src32 = (const ULONG*)src;

	if ((len & 3) != 0) return;

	len /= sizeof(ULONG);
	for (ULONG i = 0; i < len; i++) dest32[i] = src32[i];
}

static PUSBMS_CONTROLLER MspGetControllerByKey(ULONG ArcKey) {
	if (ArcKey == 0) return NULL;
	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		if (s_MassStorageDevices[i].ArcKey == ArcKey) return &s_MassStorageDevices[i];
	}
	return NULL;
}

static PUSBMS_CONTROLLER MspGetEmptyController(void) {
	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		if (s_MassStorageDevices[i].ArcKey == 0) return &s_MassStorageDevices[i];
	}
	return NULL;
}

static void MspToggleLed(void) {
	MmioWriteBase32(MEM_PHYSICAL_TO_K1(0x0d800000), 0xc0, MmioReadBase32(MEM_PHYSICAL_TO_K1(0x0d800000), 0xc0) ^ 0x20);
}

static LONG MspCtrlMsgTimeout(IOS_USB_HANDLE DeviceHandle, UCHAR RequestType, UCHAR Request, USHORT Value, USHORT Index, USHORT Length, PVOID Data, ULONG SecondsTimeout) {
	LONG Status = UlTransferControlMessageAsync(DeviceHandle, RequestType, Request, Value, Index, Length, Data);
	if (Status < 0) return Status;

	// Wait on the async operation with timeout.
	ULONG AsyncIndex = (ULONG)Status;
	PVOID Context = NULL;
	Status = -2;
	for (ULONG i = 0; i < (SecondsTimeout * 1000); i++) {
		if (PxiIopIoctlvAsyncPoll(AsyncIndex, &Status, &Context)) break;
		udelay(1000);
	}

	if (Status == -2) {
		// Cancel the control message.
		UlCancelEndpoint(DeviceHandle, USB_CANCEL_CONTROL);
		// And wait for the failure IPC response.
		while (!PxiIopIoctlvAsyncPoll(AsyncIndex, &Status, &Context)) udelay(1000);
		// Turn it into a failing status.
		Status = -2;
	}

	// Ensure the usblow context gets freed.
	UlGetPassedAsyncContext(Context);

	// And return.
	return Status;
}

static LONG MspBulkMsgTimeout(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data, ULONG SecondsTimeout) {
	LONG Status = UlTransferBulkMessageAsync(DeviceHandle, Endpoint, Length, Data);
	if (Status < 0) return Status;

	// Wait on the async operation with timeout.
	ULONG AsyncIndex = (ULONG)Status;
	PVOID Context = NULL;
	Status = -2;
	for (ULONG i = 0; i < (SecondsTimeout * 1000); i++) {
		if (PxiIopIoctlvAsyncPoll(AsyncIndex, &Status, &Context)) break;
		udelay(1000);
	}

	if (Status == -2) {
		// Cancel the endpoint message.
		UlCancelEndpoint(DeviceHandle, Endpoint);
		// And wait for the failure IPC response.
		while (!PxiIopIoctlvAsyncPoll(AsyncIndex, &Status, &Context)) udelay(1000);
		// Turn it into a failing status.
		Status = -2;
	}

	// Ensure the usblow context gets freed.
	UlGetPassedAsyncContext(Context);

	// And return.
	return Status;
}

static LONG MspReset(IOS_USB_HANDLE DeviceHandle, UCHAR Interface, UCHAR EndpointIn, UCHAR EndpointOut) {
	LONG Status = MspCtrlMsgTimeout(
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

static LONG MspSendCbw(
	PUSBMS_CONTROLLER Controller,
	UCHAR Lun,
	ULONG Length,
	UCHAR Flags,
	PUCHAR Cb,
	UCHAR CbLength,
	ULONG SecondsTimeout,
	PULONG pTag
) {
	if (CbLength == 0 || CbLength > 16) return -1;

	USBMS_CBW Cbw;
	ZeroMemory32(&Cbw, sizeof(Cbw));
	PUSBMS_CBW pCbw = PxiIopAlloc(sizeof(USBMS_CBW));
	if (pCbw == NULL) return -1;
	Cbw.Signature = CBW_SIGNATURE;
	Controller->CbwTag++;
	*pTag = Controller->CbwTag;
	Cbw.Tag = Controller->CbwTag;
	Cbw.Length = Length;
	Cbw.Flags = Flags;
	Cbw.CbLength = (CbLength > 6 ? 10 : 6);
	memcpy(Cbw.Cb, Cb, CbLength);
	memcpy32(pCbw, &Cbw, sizeof(Cbw));

	LONG Status = MspBulkMsgTimeout(
		Controller->DeviceHandle,
		Controller->EndpointOut,
		CBW_SIZE,
		pCbw,
		SecondsTimeout
	);
	PxiIopFree(pCbw);

	if (Status < 0) return Status;

	if (Status == CBW_SIZE) return 0;
	return -1;
}

static LONG MspReadCsw(
	PUSBMS_CONTROLLER Controller,
	ULONG Tag,
	PUCHAR CswStatus,
	PULONG DataResidue,
	ULONG SecondsTimeout
) {
	PUSBMS_CSW Csw = PxiIopAlloc(sizeof(USBMS_CSW));
	if (Csw == NULL) return -1;

	ZeroMemory32(Csw, sizeof(*Csw));
	LONG Status = MspBulkMsgTimeout(
		Controller->DeviceHandle,
		Controller->EndpointIn,
		CSW_SIZE,
		Csw,
		SecondsTimeout
	);

	do {
		if (Status < 0) break;
		if (Status != CSW_SIZE) {
			Status = -1;
			break;
		}

		if (Csw->Signature != CSW_SIGNATURE) {
			Status = -1;
			break;
		}
		Status = 0;
		if (CswStatus != NULL) {
			*CswStatus = Csw->Status;
		}
		if (DataResidue != NULL) {
			*DataResidue = Csw->DataResidue;
		}

		if (Csw->Tag != Tag) {
			Status = -1;
		}
	} while (false);

	PxiIopFree(Csw);
	return Status;
}

LONG MspRunScsi(
	PUSBMS_CONTROLLER Controller,
	UCHAR Lun,
	PUCHAR Buffer,
	ULONG Length,
	PUCHAR Cb,
	UCHAR CbLength,
	bool Write,
	PUCHAR CswStatus,
	PULONG CswDataResidue,
	ULONG SecondsTimeout
) {
	// Toggle the disc LED gpio.
	MspToggleLed();
	USHORT MaxSize = Controller->MaxSize;
	LONG Status = 0;
	ULONG Endpoint = (Write ? Controller->EndpointOut : Controller->EndpointIn);
	UCHAR LocalCswStatus = 0;
	ULONG LocalDataResidue = 0;
	for (UCHAR Retry = 0; Retry < USBSTORAGE_CYCLE_RETRIES; Retry++) {
		ULONG CurrLength = Length;
		PUCHAR Pointer = Buffer;
		if (Status == -2) break;
		ULONG Tag;

		Status = MspSendCbw(
			Controller, Lun,
			CurrLength,
			(Write ? CBW_OUT : CBW_IN),
			Cb,
			CbLength,
			SecondsTimeout,
			&Tag
		);

		while (CurrLength > 0 && Status >= 0) {
			ULONG RoundLength = CurrLength > MaxSize ? MaxSize : CurrLength;
			//ULONG ExtraLength = RoundLength & 7;

			// Always go through the map buffer.
			if (Write && Buffer != NULL) {
				memcpy(Controller->Buffer, Pointer, RoundLength);
			}
			Status = MspBulkMsgTimeout(
				Controller->DeviceHandle,
				Endpoint,
				RoundLength,
				Controller->Buffer,
				SecondsTimeout
			);
			ULONG BytesTransferred = 0;
			if (Status >= 0) {
				BytesTransferred = Status;
				if (!Write && Buffer != NULL) {
					memcpy(Pointer, Controller->Buffer, RoundLength);
				}
				if (BytesTransferred == RoundLength) {
					CurrLength -= BytesTransferred;
					if (Buffer != NULL) Pointer += BytesTransferred;
				}
				else Status = -1;
			}
		}

		if (Status >= 0) {
			Status = MspReadCsw(Controller, Tag, &LocalCswStatus, &LocalDataResidue, SecondsTimeout);
		}

		if (Status < 0) {
			LONG ResetStatus = MspReset(
				Controller->DeviceHandle,
				Controller->Interface,
				Controller->EndpointIn,
				Controller->EndpointOut
			);
			if (ResetStatus == -2) Status = ResetStatus;
		}
		else {
			Status = 0;
			break;
		}
	}

	if (CswStatus != NULL) *CswStatus = LocalCswStatus;
	if (CswDataResidue != NULL) *CswDataResidue = LocalDataResidue;
	// Toggle the disc LED gpio.
	MspToggleLed();
	return Status;
}

static LONG MspRunScsiMeta(
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
	if (Length > MaxSize) return -1;
	// This is a metadata request.
	// Instead of using the buffer from the controller;
	// allocate a map buffer from the IPC area.
	// This is so ioctls can be performed synchronously.
	PUCHAR MapBuffer = NULL;
	if (Length != 0) {
		MapBuffer = (PUCHAR)PxiIopAlloc(Length);
		if (MapBuffer == NULL) return -1;
	}
	LONG Status = 0;
	ULONG Endpoint = Controller->EndpointIn;
	UCHAR LocalCswStatus = 0;
	ULONG LocalDataResidue = 0;
	for (UCHAR Retry = 0; Retry < USBSTORAGE_CYCLE_RETRIES; Retry++) {
		if (Status == -2) break;
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

		if (Status >= 0 && Length != 0) {
			// Always go through the map buffer.
			Status = MspBulkMsgTimeout(
				Controller->DeviceHandle,
				Endpoint,
				Length,
				MapBuffer,
				SecondsTimeout
			);
			ULONG BytesTransferred = 0;
			if (Status >= 0) {
				if (Buffer != NULL) {
					BytesTransferred = Status;
					memcpy(Buffer, MapBuffer, BytesTransferred);
				}
				if (TransferredLength) *TransferredLength = BytesTransferred;
			}
		}

		if (Status >= 0) {
			Status = MspReadCsw(Controller, Tag, &LocalCswStatus, &LocalDataResidue, SecondsTimeout);
		}

		if (Status < 0) {
			LONG ResetStatus = MspReset(
				Controller->DeviceHandle,
				Controller->Interface,
				Controller->EndpointIn,
				Controller->EndpointOut
			);
			if (ResetStatus == -2) Status = ResetStatus;
		}
		else {
			Status = 0;
			break;
		}
	}

	if (CswStatus != NULL) *CswStatus = LocalCswStatus;
	if (CswDataResidue != NULL) *CswDataResidue = LocalDataResidue;
	if (MapBuffer != NULL) PxiIopFree(MapBuffer);
	return Status;
}

static LONG MspLowRead10(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, USHORT SectorCount, ULONG SectorShift, PVOID Buffer) {
	LONG Status;
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
		false,
		&CswStatus,
		NULL,
		10
	);
	if (Status < 0) return Status;
	if (CswStatus != 0) return -1;
	return 0;
}

static LONG MspLowWrite10(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, USHORT SectorCount, ULONG SectorShift, PVOID Buffer) {
	LONG Status;
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
		true,
		&CswStatus,
		NULL,
		10
	);
	if (Status < 0) return Status;
	if (CswStatus != 0) return -1;
	return 0;
}

static ULONG MspLowRead(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, ULONG SectorCount, ULONG SectorShift, PUCHAR Buffer) {
	USHORT CurrentRound;
	LONG Status;
	ULONG TxSectors = 0;
	while (SectorCount != 0) {
		if (SectorCount > 0xFFFF) CurrentRound = 0xFFFF;
		else CurrentRound = (USHORT)SectorCount;

		Status = MspLowRead10(Controller, Lun, SectorStart, CurrentRound, SectorShift, Buffer);
		if (Status < 0) break;

		ULONG Length = CurrentRound << SectorShift;
		SectorStart += CurrentRound;
		SectorCount -= CurrentRound;
		Buffer += Length;
		TxSectors += CurrentRound;
	}
	return TxSectors;
}

static ULONG MspLowWrite(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SectorStart, ULONG SectorCount, ULONG SectorShift, PUCHAR Buffer) {
	USHORT CurrentRound;
	LONG Status;
	ULONG TxSectors = 0;
	while (SectorCount != 0) {
		if (SectorCount > 0xFFFF) CurrentRound = 0xFFFF;
		else CurrentRound = (USHORT)SectorCount;

		Status = MspLowWrite10(Controller, Lun, SectorStart, CurrentRound, SectorShift, Buffer);
		if (Status < 0) break;

		ULONG Length = CurrentRound << SectorShift;
		SectorStart += CurrentRound;
		SectorCount -= CurrentRound;
		Buffer += Length;
		TxSectors += CurrentRound;
	}
	return TxSectors;
}


static LONG MspClearErrors(PUSBMS_CONTROLLER Controller, UCHAR Lun, ULONG SecondsTimeout) {
	LONG Status;
	UCHAR ScsiCdb[6] = { 0, 0, 0, 0, 0, 0 };
	UCHAR Sense[SCSI_SENSE_REPLY_SIZE];
	UCHAR CswStatus = 0;

	ScsiCdb[0] = SCSI_TEST_UNIT_READY;

	Status = MspRunScsiMeta(Controller, Lun, NULL, 0, ScsiCdb, sizeof(ScsiCdb), NULL, &CswStatus, NULL, SecondsTimeout);
	if (Status < 0) return Status;

	if (CswStatus) {
		ScsiCdb[0] = SCSI_REQUEST_SENSE;
		ScsiCdb[1] = Lun << 5;
		ScsiCdb[4] = sizeof(Sense);
		ULONG Transferred = 0;
		Status = MspRunScsiMeta(Controller, Lun, Sense, sizeof(Sense), ScsiCdb, sizeof(ScsiCdb), &Transferred, NULL, NULL, SecondsTimeout);
		if (Status < 0) return Status;
		if (Transferred < sizeof(Sense)) return -1;

		UCHAR SenseErr = Sense[2] & 0xF;
		if (SenseErr == SCSI_SENSE_NOT_READY) return -1;
		if (SenseErr == SCSI_SENSE_MEDIUM_ERROR) return -1;
		if (SenseErr == SCSI_SENSE_HARDWARE_ERROR) return -1;
		if (SenseErr == SCSI_SENSE_UNIT_ATTENTION) return -3;
	}

	return Status;
}

static bool MspCapacityIsFloppy(PSCSI_FORMAT_CAPACITY_ENTRY Entries, UCHAR Count) {
	for (ULONG i = 0; i < Count; i++) {
		ULONG NumberOfBlocks = Entries[i].NumberOfBlocks;
		ULONG BlockLength = (Entries[i].BlockLengthHigh << 16) |
			Entries[i].BlockLengthLow;
		if (BlockLength == 1024) {
			if (NumberOfBlocks == (77 * 2 * 8)) {
				// 1.25MB (PC-98, etc)
				return true;
			}
			continue;
		}
		if (BlockLength != 512) continue;
		// for 512-byte sectors, allow everything that NT5 usbmass.sys does;
		// assume that usbmass.sys supports it because some USB floppy drive does.
		switch (NumberOfBlocks) {
		case  80 * 2 * 8:  // 640 KB (5.25"?!)
		case  80 * 2 * 9:  // 720 KB
		case  80 * 2 * 15: // 1.2 MB
		case  80 * 2 * 18: // 1.44MB

		case 963 * 8 * 32: // 120MB, SuperDisk LS-120

		case 890 * 13 * 34: // 200MB, Sony HiFD
			return true;
		}
	}
	return false;
}

static LONG MspGetDiskType(PUSBMS_CONTROLLER Controller, UCHAR Lun, PUSBMS_DISK_TYPE DiskType, ULONG SecondsTimeout) {
	// Do SCSI Inquiry, get 36 bytes
	LONG Status;
	UCHAR Inquiry[36];
	UCHAR ScsiCdb[6] = { 0 };
	ScsiCdb[0] = SCSI_INQUIRY;
	ScsiCdb[1] = Lun << 5;
	ScsiCdb[4] = sizeof(Inquiry);
	for (int Retry = 0; Retry < 2; Retry++) {
		ZeroMemory32(Inquiry, sizeof(Inquiry));
		ULONG Transferred = 0;
		Status = MspRunScsiMeta(Controller, Lun, Inquiry, sizeof(Inquiry), ScsiCdb, sizeof(ScsiCdb), &Transferred, NULL, NULL, SecondsTimeout);
		if (Status < 0) continue;
		if (Transferred < sizeof(Inquiry)) {
			Status = -1;
			continue;
		}
		break;
	}
	if (Status < 0) return Status;

	BOOLEAN RemovableDisk = (Inquiry[1] & 0x80) != 0;
	UCHAR DeviceType = Inquiry[0] & 0x1f;
	if (DeviceType == 5) {
		*DiskType = USBMS_DISK_CDROM;
		return 0;
	}
	if (DeviceType != 0) {
		*DiskType = USBMS_DISK_UNKNOWN;
		return 0;
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
	ZeroMemory32(FormatCapsBuf, sizeof(FormatCapsBuf));
	ZeroMemory32(ScsiCdb12, sizeof(ScsiCdb12));
	ScsiCdb12[0] = SCSI_READ_FORMAT_CAPACITY;
	ScsiCdb12[1] = Lun << 5;
	ScsiCdb12[8] = sizeof(FormatCapsBuf);
	ULONG TransferredBytes = 0;
	Status = MspRunScsiMeta(Controller, Lun, FormatCapsBuf, sizeof(FormatCapsBuf), ScsiCdb12, sizeof(ScsiCdb12), &TransferredBytes, NULL, NULL, SecondsTimeout);
	if (Status < 0) return 0;
	if (TransferredBytes < sizeof(SCSI_FORMAT_CAPACITY_LIST)) return 0;
	PSCSI_FORMAT_CAPACITY_LIST List = (PSCSI_FORMAT_CAPACITY_LIST)
		FormatCapsBuf;
	if (List->Length == 0) return 0;
	if ((List->Length % sizeof(SCSI_FORMAT_CAPACITY_ENTRY)) != 0) return 0;

	ULONG Count = TransferredBytes - sizeof(SCSI_FORMAT_CAPACITY_ENTRY);
	if (Count >= List->Length) Count = List->Length;
	Count /= sizeof(SCSI_FORMAT_CAPACITY_ENTRY);

	if (MspCapacityIsFloppy(List->Entries, Count)) {
		*DiskType = USBMS_DISK_FLOPPY;
	}

	return 0;
}

static LONG MspIsWriteProtect(PUSBMS_CONTROLLER Controller, UCHAR Lun, bool* WriteProtect, ULONG SecondsTimeout) {
	LONG Status;
	UCHAR ScsiCdb6[6] = { 0 };
	ScsiCdb6[0] = SCSI_MODE_SENSE;
	ScsiCdb6[2] = 0x3F;
	ScsiCdb6[4] = 192;
	UCHAR ModeSense[192];

	ULONG Transferred = 0;
	Status = MspRunScsiMeta(Controller, Lun, ModeSense, sizeof(ModeSense), ScsiCdb6, sizeof(ScsiCdb6), &Transferred, NULL, NULL, SecondsTimeout);

	if (Status < 0) return Status;

	if (Transferred < 4) {
		// try again
		Status = MspRunScsiMeta(Controller, Lun, ModeSense, sizeof(ModeSense), ScsiCdb6, sizeof(ScsiCdb6), &Transferred, NULL, NULL, SecondsTimeout);

		if (Status < 0) return Status;
		if (Transferred < 4) return -1;
	}

	*WriteProtect = (ModeSense[2] & 0x80) != 0;
	return 0;
}

static LONG MspReadCapacity(PUSBMS_CONTROLLER Controller, UCHAR Lun, PULONG SectorSize, PULONG SectorCount, ULONG SecondsTimeout) {
	LONG Status;
	UCHAR ScsiCdb10[10] = { 0 };
	ScsiCdb10[0] = SCSI_READ_CAPACITY;
	ScsiCdb10[1] = (Lun << 5);
	SCSI_READ_CAPACITY_RES ReadCapacity;

	ULONG Transferred = 0;
	Status = MspRunScsiMeta(Controller, Lun, (PUCHAR)&ReadCapacity, sizeof(ReadCapacity), ScsiCdb10, sizeof(ScsiCdb10), &Transferred, NULL, NULL, SecondsTimeout);
	if (Status < 0) return Status;
	
	if (SectorCount != NULL) *SectorCount = ReadCapacity.SectorCount;
	if (SectorSize != NULL) *SectorSize = ReadCapacity.SectorSize;
	return 0;
}

static LONG MspCheckVerify(PUSBMS_CONTROLLER Controller, UCHAR Lun) {
	if (Lun >= Controller->NumberOfLuns) return -1;
	PUSBMS_LUN LunObj = &Controller->Luns[Lun];
	Lun = LunObj->RealLun;
	// Do clear errors. Use 20 second timeout just in case.
	LONG Status = MspClearErrors(Controller, Lun, 20);
	LONG CapStatus;

	bool IsCdRom = LunObj->DiskType == USBMS_DISK_CDROM;
	if (Status == -3) {
		// Medium changed. Go again, allow media to spin up, ignore result.
		Status = MspClearErrors(Controller, Lun, 20);
		// Read the drive capacity.
		ULONG SectorCount = 0;
		ULONG SectorSize = 0;
		LONG CapStatus = MspReadCapacity(Controller, Lun, &SectorSize, &SectorCount, 20);
		LunObj->DriveNotReady = (CapStatus < 0);
		if (CapStatus < 0 || SectorSize == 0) {
			if (IsCdRom) {
				SectorCount = 0x7fffffff;
				SectorSize = 2048;
			}
			else {
				SectorCount = 0;
				SectorSize = 512;
			}
		}
		// set the new sector-count/size
		LunObj->SectorSize = SectorSize;
		LunObj->SectorCount = SectorCount;
		LunObj->SectorShift = __builtin_ctz(SectorSize);
	}
	// Run mode sense to get write protect status.
	bool WriteProtect = IsCdRom;
	if (!WriteProtect) {
		CapStatus = MspIsWriteProtect(Controller, Lun, &WriteProtect, USBSTORAGE_TIMEOUT);
		if (Status < 0) WriteProtect = false;
	}
	LunObj->WriteProtected = WriteProtect;
	return Status;
}

static LONG MspInitDevice(
	IOS_USB_HANDLE DeviceHandle,
	ULONG ArcKey,
	PVOID UsbBuffer,
	PULONG UsbOffset
) {
	// Open device.
	LONG Status = UlOpenDevice(DeviceHandle);
	if (Status < 0) return Status;

	do {
		// Get device descriptors.
		USB_DEVICE_DESC Descriptors;
		Status = UlGetDescriptors(DeviceHandle, &Descriptors);
		if (Status < 0) break;

		if (Descriptors.Device.bNumConfigurations == 0) {
			Status = -1;
			break;
		}

		PUSB_CONFIGURATION Config = &Descriptors.Config;
		if (Config->bNumInterfaces == 0) {
			Status = -1;
			break;
		}
		PUSB_INTERFACE Interface = &Descriptors.Interface;
		if (Interface->bInterfaceClass != USB_CLASS_MASS_STORAGE) {
			Status = -1;
			break;
		}
		if (Interface->bInterfaceProtocol != MASS_STORAGE_BULK_ONLY) {
			Status = -1;
			break;
		}
		if (Interface->bNumEndpoints < 2) {
			Status = -1;
			break;
		}

		bool IsUsb2 = false;
		UCHAR EndpointIn = 0, EndpointOut = 0;
		for (ULONG i = 0; i < Interface->bNumEndpoints; i++) {
			PUSB_ENDPOINT Endpoint = &Descriptors.Endpoints[i];
			if (Endpoint->bmAttributes != USB_ENDPOINT_BULK) continue;
			if ((Endpoint->bEndpointAddress & USB_ENDPOINT_IN) != 0) {
				EndpointIn = Endpoint->bEndpointAddress;
			}
			else {
				EndpointOut = Endpoint->bEndpointAddress;
				IsUsb2 = Endpoint->wMaxPacketSize > 64;
			}
		}

		if (EndpointIn == 0 || EndpointOut == 0) {
			Status = -1;
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
		if (!IsUsb2) {
			MspReset(DeviceHandle, Interface->bInterfaceNumber, EndpointIn, EndpointOut);
		}

		PUCHAR pMaxLun = (PUCHAR)PxiIopAlloc(sizeof(UCHAR));
		if (pMaxLun == NULL) {
			Status = -1;
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
		PxiIopFree(pMaxLun);
		if (Status == -2) {
			Status = -1;
			break;
		}
		if (Status < 0) MaxLun = 1;

		if (MaxLun > USBMS_MAX_LUNS) MaxLun = USBMS_MAX_LUNS;

		// This device is working USB mass storage device.
		// We can now do SCSI inquiry and read capacity on all LUNs;
		// and create disk device for all LUNs.

		// First, create the controller for this device.
		PUSBMS_CONTROLLER Extension = MspGetEmptyController();
		if (Extension == NULL) {
			Status = -1;
			break;
		}

		// Fill in the parameters.
		Extension->DeviceHandle = DeviceHandle;
		Extension->ArcKey = ArcKey;
		Extension->EndpointIn = EndpointIn;
		Extension->EndpointOut = EndpointOut;
		Extension->Interface = Interface->bInterfaceNumber;
		Extension->MaxSize = IsUsb2 ? MAX_TRANSFER_SIZE_V2 : MAX_TRANSFER_SIZE_V1;
		Extension->NumberOfLuns = MaxLun;
		Extension->Buffer = (PVOID)(*UsbOffset + (ULONG)UsbBuffer);
		*UsbOffset += Extension->MaxSize;

		bool HasValidLun = false;
		ULONG LunArr = 0;
		for (ULONG lun = 0; lun < MaxLun; lun++) {
			USBMS_DISK_TYPE DiskType = USBMS_DISK_UNKNOWN;

			Status = MspGetDiskType(Extension, lun, &DiskType, USBSTORAGE_TIMEOUT);
			if (Status < 0) continue;

			Extension->Luns[LunArr].DiskType = DiskType;

			// Get the drive capacity, if needed.
			ULONG SectorSize = 0;
			ULONG SectorCount = 0;
			Status = MspReadCapacity(Extension, lun, &SectorSize, &SectorCount, USBSTORAGE_TIMEOUT);
			if (Status < 0 && DiskType == USBMS_DISK_OTHER_FIXED) continue;

			// Set up the current LUN.
			Extension->Luns[LunArr].SectorSize = SectorSize;
			Extension->Luns[LunArr].SectorCount = SectorCount;
			Extension->Luns[LunArr].SectorShift = __builtin_ctz(SectorSize);

			// Obtain write-protect/etc status.
			Extension->Luns[LunArr].RealLun = lun;
			Extension->Luns[LunArr].LunValid = true;
			Status = MspCheckVerify(Extension, lun);
			if (Status < 0) {
				Extension->Luns[LunArr].LunValid = false;
				continue;
			}

			LunArr++;
			HasValidLun = true;
		}
		Extension->NumberOfLuns = LunArr;
		if (HasValidLun) Status = 0;
	} while (false);
	if (Status < 0) UlCloseDevice(DeviceHandle);
	//if (!NT_SUCCESS(Status)) UlCloseDevice(DeviceHandle);
	return Status;
}

bool UlmsInit(void) {
	// Get device list.

	memset(s_MassStorageDevices, 0, sizeof(s_MassStorageDevices));
	PIOS_USB_DEVICE_ENTRY Entries = (PIOS_USB_DEVICE_ENTRY)
		PxiIopAlloc(sizeof(IOS_USB_DEVICE_ENTRY_MAX));
	if (Entries == NULL) return false;

	ZeroMemory32(Entries, sizeof(IOS_USB_DEVICE_ENTRY_MAX));

	UCHAR NumVen = 0;
	UlGetDeviceList(Entries, USB_COUNT_DEVICES, USB_CLASS_MASS_STORAGE, &NumVen);

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
		LONG Status = MspInitDevice(Entries[i].DeviceHandle, ArcKey, MEM_PHYSICAL_TO_K1(USB_BUFFER_PHYS_START), &UsbOffset);
		if (Status >= 0) {
			// successful, device was created.
			NumMs++;
		}
	}

	PxiIopFree(Entries);

	// Return success, even if no devices were found.
	return true;
}

void UlmsFinalise(void) {
	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		if (s_MassStorageDevices[i].ArcKey == 0) continue;
		UlClearHalt(s_MassStorageDevices[i].DeviceHandle);
		UlCloseDevice(s_MassStorageDevices[i].DeviceHandle);
		s_MassStorageDevices[i].ArcKey = 0;
	}
}

void UlmsGetDevices(PUSBMS_DEVICES Devices) {
	Devices->DeviceCount = 0;
	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		if (s_MassStorageDevices[i].ArcKey == 0) continue;
		Devices->ArcKey[Devices->DeviceCount] = s_MassStorageDevices[i].ArcKey;
		Devices->DeviceCount++;
	}
}

PUSBMS_CONTROLLER UlmsGetController(ULONG ArcKey) {
	return MspGetControllerByKey(ArcKey);
}

ULONG UlmsGetLuns(PUSBMS_CONTROLLER Controller) {
	if (Controller == NULL) return 0;
	return Controller->NumberOfLuns;
}

ULONG UlmsGetSectorSize(PUSBMS_CONTROLLER Controller, ULONG Lun) {
	if (Controller == NULL) return 0;
	if (Lun >= Controller->NumberOfLuns) return 0;
	return Controller->Luns[Lun].SectorSize;
}

ULONG UlmsGetSectorCount(PUSBMS_CONTROLLER Controller, ULONG Lun) {
	if (Controller == NULL) return 0;
	if (Lun >= Controller->NumberOfLuns) return 0;
	return Controller->Luns[Lun].SectorCount;
}

ULONG UlmsReadSectors(PUSBMS_CONTROLLER Controller, ULONG Lun, ULONG Sector, ULONG NumSector, PVOID Buffer) {
	if (Controller == NULL) return 0;
	if (Lun >= Controller->NumberOfLuns) return 0;
	PUSBMS_LUN LunObj = &Controller->Luns[Lun];
	if (!LunObj->LunValid) return 0;
	Lun = LunObj->RealLun;

	return MspLowRead(Controller, Lun, Sector, NumSector, LunObj->SectorShift, Buffer);
}

ULONG UlmsWriteSectors(PUSBMS_CONTROLLER Controller, ULONG Lun, ULONG Sector, ULONG NumSector, const void* Buffer) {
	if (Controller == NULL) return 0;
	if (Lun >= Controller->NumberOfLuns) return 0;
	PUSBMS_LUN LunObj = &Controller->Luns[Lun];
	if (!LunObj->LunValid) return 0;
	Lun = LunObj->RealLun;

	return MspLowWrite(Controller, Lun, Sector, NumSector, LunObj->SectorShift, Buffer);
}
