#include <stddef.h>
#include <memory.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <errno.h>
#include "arc.h"
#include "arcdevice.h"
#include "arcconfig.h"
#include "arcio.h"
#include "arcfs.h"
#include "arcenv.h"

#include "ios_usb_ms.h" // USB mass storage, used "raw"
#include "fatfs/ff.h" // FatFS used for accessing disk images on other storage devices
#include "fatfs/diskio.h" // FatFS diskio

// ARC firmware support for disks:
// USB mass storage (raw devices)
// EXI bus devices and wii sd slot (disk images on those devices)

enum {
	MAXIMUM_SECTOR_SIZE = 2048,
	USB_CLASS_MASS_STORAGE = 0x08
};

typedef struct _USB_DEVICE_MOUNT_TABLE {
	ULONG Address; // USB device address. Guaranteed by the USB stack to be non-zero.
	ULONG SectorSize; // Sector size.
	ULONG ReferenceCount; // Reference count as callers could mount same device multiple times, for ro/rw/wo.
	PUSBMS_CONTROLLER Controller;
} USB_DEVICE_MOUNT_TABLE, *PUSB_DEVICE_MOUNT_ENTRY;



// Mount table.
static USB_DEVICE_MOUNT_TABLE s_MountTable[32] = { 0 };

// ARC path for the controller devices.
// disk(x)cdrom(x) is invalid except for x86 el torito, says osloader
// use scsi(0) for USB, because USB mass storage is SCSI lol
static char s_UsbControllerPath[] = "scsi(0)";
static char s_ExiControllerPath[] = "multi(0)";
static char s_IossdmcControllerPath[] = "multi(1)";

static char s_UsbComponentName[] = "IOS_USB";
static char s_IossdmcComponentName[] = "IOS_SDMC";
static char s_ExiComponentName[] = "EXI";

// Determine disk type by ARC device path.
// Returns diskio enumeration value, or (DEV_IOS_SDMC + 1) for USB, or 0xFFFFFFFF for unknown.
ULONG ArcDiskGetDiskType(const char* DevicePath) {
	if (memcmp(DevicePath, s_UsbControllerPath, sizeof(s_UsbControllerPath) - 1) == 0) return (DEV_IOS_SDMC + 1);
	if (memcmp(DevicePath, s_IossdmcControllerPath, sizeof(s_IossdmcControllerPath) - 1) == 0) return DEV_IOS_SDMC;
	if (memcmp(DevicePath, s_ExiControllerPath, sizeof(s_ExiControllerPath) - 1) != 0) return 0xFFFFFFFFul;


	PCHAR Path = &DevicePath[sizeof(s_ExiControllerPath) - 1];
	// Next device must be disk(x)
	ULONG Index = 0;
	if (!ArcDeviceParse(&Path, DiskController, &Index)) {
		return 0xFFFFFFFFul;
	}

	// Got the index, make sure this is good
	if (Index >= 4) return 0xFFFFFFFFul;
	return Index;
}

// Mount a usb device by vid/pid.
static ARC_STATUS UsbDiskMount(ULONG Address, PUSB_DEVICE_MOUNT_ENTRY* Handle) {
	// Check the mount table to see if this vid/pid is mounted.
	for (ULONG i = 0; i < sizeof(s_MountTable) / sizeof(s_MountTable[0]); i++) {
		if (s_MountTable[i].Address == Address) {
			// It's attached, return the existing handle.
			ULONG NewRefCount = s_MountTable[i].ReferenceCount + 1;
			if (NewRefCount == 0) {
				// Reference count overflow
				return _EBUSY;
			}
			s_MountTable[i].ReferenceCount = NewRefCount;
			*Handle = &s_MountTable[i];
			return _ESUCCESS;
		}
	}

	// Grab the usb controller.
	PUSBMS_CONTROLLER Controller = UlmsGetController(Address);
	if (Controller == NULL) return _ENODEV;
	if (UlmsGetLuns(Controller) == 0) return _ENODEV;

	// Find an empty slot in the mount table
	for (ULONG i = 0; i < sizeof(s_MountTable) / sizeof(s_MountTable[0]); i++) {
		if (s_MountTable[i].Address != 0) continue;
		s_MountTable[i].Address = Address;
		s_MountTable[i].ReferenceCount = 1;
		s_MountTable[i].Controller = Controller;
		s_MountTable[i].SectorSize = UlmsGetSectorSize(Controller, 0);
		*Handle = &s_MountTable[i];
		return _ESUCCESS;
	}

	return _ENODEV;
}

// Unmount a usb device.
static ARC_STATUS UsbDiskUnMount(PUSB_DEVICE_MOUNT_ENTRY Handle) {
	// Ensure this handle looks good.
	ULONG i = (size_t)Handle - (size_t)&s_MountTable[0];
	if (i >= sizeof(s_MountTable)) return _EBADF;

	ULONG NewRefCount = Handle->ReferenceCount - 1;
	if (NewRefCount == 0) {
		// Wipe the table entry.
		memset(Handle, 0, sizeof(*Handle));
	}
	else Handle->ReferenceCount = NewRefCount;
	return _ESUCCESS;
}

static ARC_STATUS DeblockerRead(ULONG FileId, PVOID Buffer, ULONG Length, PULONG Count);
static ARC_STATUS DeblockerWrite(ULONG FileId, PVOID Buffer, ULONG Length, PULONG Count);
static ARC_STATUS DeblockerSeek(ULONG FileId, PLARGE_INTEGER Offset, SEEK_MODE SeekMode);

static ARC_STATUS UsbDiskOpen(PCHAR OpenPath, OPEN_MODE OpenMode, PULONG FileId);
static ARC_STATUS UsbDiskClose(ULONG FileId);
static ARC_STATUS UsbDiskArcMount(PCHAR MountPath, MOUNT_OPERATION Operation);
static ARC_STATUS UsbDiskRead(PARC_FILE_TABLE FileEntry, ULONG StartSector, ULONG CountSectors, PVOID Buffer);
static ARC_STATUS UsbDiskWrite(PARC_FILE_TABLE FileEntry, ULONG StartSector, ULONG CountSectors, PVOID Buffer);
static ARC_STATUS UsbDiskGetReadStatus(ULONG FileId);
static ARC_STATUS UsbDiskGetFileInformation(ULONG FileId, PFILE_INFORMATION FileInfo);

static ARC_STATUS ImgOpen(PCHAR OpenPath, OPEN_MODE OpenMode, PULONG FileId);
static ARC_STATUS ImgClose(ULONG FileId);
static ARC_STATUS ImgMount(PCHAR MountPath, MOUNT_OPERATION Operation);
static ARC_STATUS ImgSeek(ULONG FileId, PLARGE_INTEGER Offset, SEEK_MODE SeekMode);
static ARC_STATUS ImgRead(PARC_FILE_TABLE FileEntry, ULONG StartSector, ULONG CountSectors, PVOID Buffer);
static ARC_STATUS ImgWrite(PARC_FILE_TABLE FileEntry, ULONG StartSector, ULONG CountSectors, PVOID Buffer);
static ARC_STATUS ImgGetReadStatus(ULONG FileId);
static ARC_STATUS ImgGetFileInformation(ULONG FileId, PFILE_INFORMATION FileInfo);

// USB controller device vectors.
static const DEVICE_VECTORS UsbDiskVectors = {
	.Open = UsbDiskOpen,
	.Close = UsbDiskClose,
	.Mount = UsbDiskArcMount,
	.Read = DeblockerRead,
	.Write = DeblockerWrite,
	.Seek = DeblockerSeek,
	.GetReadStatus = UsbDiskGetReadStatus,
	.GetFileInformation = UsbDiskGetFileInformation,
	.SetFileInformation = NULL,
	.GetDirectoryEntry = NULL
};

// USB controller device functions.

static ARC_STATUS UsbDiskGetSectorSize(ULONG FileId, PULONG SectorSize) {
	// Get the file table entry.
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EFAULT;

	// Get the mount table entry.
	PUSB_DEVICE_MOUNT_ENTRY MountEntry = FileEntry->u.DiskContext.DeviceMount;
	if (MountEntry == NULL) return _EFAULT;

	// Retrieve the sector size from it
	*SectorSize = MountEntry->SectorSize;
	return _ESUCCESS;
}

static ARC_STATUS UsbDiskOpen(PCHAR OpenPath, OPEN_MODE OpenMode, PULONG FileId) {
	// Ensure the path starts with s_ControllerPath
	if (memcmp(OpenPath, s_UsbControllerPath, sizeof(s_UsbControllerPath) - 1) != 0) return _ENODEV;
	PCHAR DevicePath = &OpenPath[sizeof(s_UsbControllerPath) - 1];
	// Next device must be disk(x) or cdrom(x)
	ULONG UsbId = 0;
	bool IsCdRom = false;
	if (!ArcDeviceParse(&DevicePath, DiskController, &UsbId)) {
		// Not disk(x), check for cdrom(x)
		IsCdRom = true;
		if (!ArcDeviceParse(&DevicePath, CdromController, &UsbId)) {
			// Not cdrom either, can't handle this device path
			return _ENODEV;
		}
	}
	else {
		// Next device must be rdisk(0)
		ULONG MustBeZero = 0;
		if (!ArcDeviceParse(&DevicePath, DiskPeripheral, &MustBeZero)) {
			// not found
			return _ENODEV;
		}
		if (MustBeZero != 0) return _ENODEV;
	}

	// If this is a cdrom, it can only be mounted ro.
	if (IsCdRom && OpenMode != ArcOpenReadOnly) return _EACCES;

	bool IncludesPartition = false;
	ULONG PartitionNumber = 0;
	// Does the caller want a partition?
	if (*DevicePath != 0) {
		if (!ArcDeviceParse(&DevicePath, PartitionEntry, &PartitionNumber)) {
			// osloader expects fdisk(x)
			ULONG DontCare;
			if (!ArcDeviceParse(&DevicePath, FloppyDiskPeripheral, &DontCare))
				return _ENODEV;
			// which should be the last device, but...
			if (*DevicePath != 0) {
				// partition is still technically valid here.
				if (!ArcDeviceParse(&DevicePath, PartitionEntry, &PartitionNumber))
					return _ENODEV;

				// partition number here must be zero
				if (PartitionNumber != 0) return _ENODEV;
			}
		}
		else {
			// partition 0 means whole disk
			IncludesPartition = PartitionNumber != 0;
		}
	}

	// Get the file table entry.
	PARC_FILE_TABLE FileEntry = ArcIoGetFileForOpen(*FileId);
	if (FileEntry == NULL) return _EFAULT;
	// Zero the disk context.
	memset(&FileEntry->u.DiskContext, 0, sizeof(FileEntry->u.DiskContext));

	// It's now known if this is a disk or cdrom (ie, whether to use FAT or ISO9660 filesystem driver)
	// Mount the usb device, if required.
	PUSB_DEVICE_MOUNT_ENTRY Handle;
	ARC_STATUS Status = UsbDiskMount(UsbId, &Handle);
	if (ARC_FAIL(Status)) return Status;

	// Stash the mount handle into the file table.
	FileEntry->u.DiskContext.DeviceMount = Handle;
	FileEntry->u.DiskContext.MaxSectorTransfer = 0xFFFF;
	// Stash the GetSectorSize ptr into the file table.
	FileEntry->GetSectorSize = UsbDiskGetSectorSize;
	FileEntry->ReadSectors = UsbDiskRead;
	FileEntry->WriteSectors = UsbDiskWrite;

	ULONG PartitionSectors = UlmsGetSectorCount(Handle->Controller, 0);
	ULONG PartitionSector = 0;
	// Set it up for the whole disk so ArcFsPartitionObtain can work.
	FileEntry->u.DiskContext.SectorStart = PartitionSector;
	FileEntry->u.DiskContext.SectorCount = PartitionSectors;
	if (IncludesPartition) {
		// Mark the file as open so ArcFsPartitionObtain can work.
		FileEntry->Flags.Open = 1;
		Status = ArcFsPartitionObtain(FileEntry->DeviceEntryTable, *FileId, PartitionNumber, Handle->SectorSize, &PartitionSector, &PartitionSectors);
		FileEntry->Flags.Open = 0;
		if (ARC_FAIL(Status)) {
			PartitionSector = 0;
			PartitionSectors = Handle->SectorSize;
		}
	}
	FileEntry->u.DiskContext.SectorStart = PartitionSector;
	FileEntry->u.DiskContext.SectorCount = PartitionSectors;
	return _ESUCCESS;
}
static ARC_STATUS UsbDiskClose(ULONG FileId) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;
	// Unmount the USB device.
	PUSB_DEVICE_MOUNT_ENTRY MountEntry = FileEntry->u.DiskContext.DeviceMount;
	return UsbDiskUnMount(MountEntry);
}
static ARC_STATUS UsbDiskArcMount(PCHAR MountPath, MOUNT_OPERATION Operation) { return _EINVAL; }
static ARC_STATUS DeblockerSeek(ULONG FileId, PLARGE_INTEGER Offset, SEEK_MODE SeekMode) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;
	
	ULONG SectorSize;
	ARC_STATUS Status = FileEntry->GetSectorSize(FileId, &SectorSize);
	if (ARC_FAIL(Status)) return Status;

	switch (SeekMode) {
	case SeekRelative:
		FileEntry->Position += Offset->QuadPart;
		break;
	case SeekAbsolute:
		FileEntry->Position = Offset->QuadPart;
		break;
	default:
		return _EINVAL;
	}

	LARGE_INTEGER SizeInBytes;
	SizeInBytes.QuadPart = FileEntry->u.DiskContext.SectorCount;
	SizeInBytes.QuadPart *= SectorSize;
	if (FileEntry->Position > SizeInBytes.QuadPart) FileEntry->Position = SizeInBytes.QuadPart;

	return _ESUCCESS;
}

static BYTE s_TemporaryBuffer[MAXIMUM_SECTOR_SIZE + 128];

static ARC_STATUS DeblockerRead(ULONG FileId, PVOID Buffer, ULONG Length, PULONG Count) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;
	ULONG SectorSize;
	ARC_STATUS Status = FileEntry->GetSectorSize(FileId, &SectorSize);
	if (ARC_FAIL(Status)) return Status;
	ULONG SectorStart = FileEntry->u.DiskContext.SectorStart;
	ULONG SectorCount = FileEntry->u.DiskContext.SectorCount;
	ULONG TransferCount;

	PBYTE LocalPointer = (PVOID)(((ULONG)(&s_TemporaryBuffer[DCACHE_LINE_SIZE - 1])) & ~(DCACHE_LINE_SIZE - 1));

	*Count = 0;

	// If the current position is not at a sector boundary, read the first sector seperately.
	ULONG Offset = FileEntry->Position & (SectorSize - 1);
	
	// Hardcode the two most common sector sizes to avoid expensive divdi3 calls
	ULONG CurrentSector = 0;
	if (SectorSize == 0x200) CurrentSector = (FileEntry->Position - Offset) / 0x200;
	else if (SectorSize == 0x800) CurrentSector = (FileEntry->Position - Offset) / 0x800;
	else CurrentSector = (FileEntry->Position - Offset) / SectorSize;
	
	if (Offset != 0) {
		Status = FileEntry->ReadSectors(FileEntry, CurrentSector + SectorStart, 1, LocalPointer);
		if (ARC_FAIL(Status)) {
			return Status;
		}

		ULONG Limit;
		if ((SectorSize - Offset) > Length) Limit = Length;
		else Limit = SectorSize - Offset;
		memcpy(Buffer, &LocalPointer[Offset], Limit);
		Buffer = (PVOID)((size_t)Buffer + Limit);
		Length -= Limit;
		*Count += Limit;
		FileEntry->Position += Limit;
		CurrentSector++;
	}

	// At a sector boundary, so read as many sectors as possible.
	ULONG BytesToTransfer = Length & (~(SectorSize - 1));
	while (BytesToTransfer != 0) {
		// Low-level driver only supports transfer of up to 64K sectors.
		ULONG SectorsToTransfer = BytesToTransfer / SectorSize;
		if (SectorsToTransfer > FileEntry->u.DiskContext.MaxSectorTransfer) SectorsToTransfer = FileEntry->u.DiskContext.MaxSectorTransfer;

		if ((CurrentSector + SectorsToTransfer) > SectorCount) {
			SectorsToTransfer = SectorCount - CurrentSector;
		}

		if (SectorsToTransfer == 0) break;

		Status = FileEntry->ReadSectors(FileEntry, CurrentSector + SectorStart, SectorsToTransfer, Buffer);
		if (ARC_FAIL(Status)) {
			return Status;
		}

		ULONG Limit = SectorsToTransfer * SectorSize;
		*Count += Limit;
		Length -= Limit;
		Buffer = (PVOID)((size_t)Buffer + Limit);
		BytesToTransfer -= Limit;
		FileEntry->Position += Limit;
		CurrentSector += SectorsToTransfer;
	}

	// If there's any data left to read, read the last sector.
	if (Length != 0) {
		Status = FileEntry->ReadSectors(FileEntry, CurrentSector + SectorStart, 1, LocalPointer);
		if (ARC_FAIL(Status)) {
			return Status;
		}

		memcpy(Buffer, LocalPointer, Length);
		*Count += Length;
		FileEntry->Position += Length;
	}

	return _ESUCCESS;
}

static ARC_STATUS UsbDiskRead(PARC_FILE_TABLE FileEntry, ULONG StartSector, ULONG CountSectors, PVOID Buffer) {
	PUSB_DEVICE_MOUNT_ENTRY MountEntry = FileEntry->u.DiskContext.DeviceMount;
	if (MountEntry == NULL) return _EBADF;

	ULONG ReadSectors = UlmsReadSectors(MountEntry->Controller, 0, StartSector, CountSectors, Buffer);
	if (ReadSectors != CountSectors) return _EIO;
	return _ESUCCESS;
}

static ARC_STATUS DeblockerWrite(ULONG FileId, PVOID Buffer, ULONG Length, PULONG Count) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;
	ULONG SectorSize;
	ARC_STATUS Status = FileEntry->GetSectorSize(FileId, &SectorSize);
	if (ARC_FAIL(Status)) return Status;
	ULONG SectorStart = FileEntry->u.DiskContext.SectorStart;
	ULONG SectorCount = FileEntry->u.DiskContext.SectorCount;
	ULONG TransferCount;

	PBYTE LocalPointer = (PVOID)(((ULONG)(&s_TemporaryBuffer[DCACHE_LINE_SIZE - 1])) & ~(DCACHE_LINE_SIZE - 1));

	*Count = 0;

	// If the current position is not at a sector boundary, read the first sector seperately, replace the data, and write back to disk
	ULONG Offset = FileEntry->Position & (SectorSize - 1);
	
	// Hardcode the two most common sector sizes to avoid expensive divdi3 calls
	ULONG CurrentSector = 0;
	if (SectorSize == 0x200) CurrentSector = (FileEntry->Position - Offset) / 0x200;
	else if (SectorSize == 0x800) CurrentSector = (FileEntry->Position - Offset) / 0x800;
	else CurrentSector = (FileEntry->Position - Offset) / SectorSize;
	
	if (Offset != 0) {
		Status = FileEntry->ReadSectors(FileEntry, CurrentSector + SectorStart, 1, LocalPointer);
		if (ARC_FAIL(Status)) {
			return Status;
		}

		ULONG Limit;
		if ((SectorSize - Offset) > Length) Limit = Length;
		else Limit = SectorSize - Offset;
		memcpy(&LocalPointer[Offset], Buffer, Limit);

		// Write the sector.
		Status = FileEntry->WriteSectors(FileEntry, CurrentSector + SectorStart, 1, LocalPointer);
		if (ARC_FAIL(Status)) {
			return Status;
		}

		Buffer = (PVOID)((size_t)Buffer + Limit);
		Length -= Limit;
		*Count += Limit;
		FileEntry->Position += Limit;
		CurrentSector++;
	}

	// At a sector boundary, so write as many sectors as possible.
	ULONG BytesToTransfer = Length & (~(SectorSize - 1));
	while (BytesToTransfer != 0) {
		// Low-level driver only supports transfer of up to 64K sectors.
		ULONG SectorsToTransfer = BytesToTransfer / SectorSize;
		if (SectorsToTransfer > FileEntry->u.DiskContext.MaxSectorTransfer) SectorsToTransfer = FileEntry->u.DiskContext.MaxSectorTransfer;

		if ((CurrentSector + SectorsToTransfer) > SectorCount) {
			SectorsToTransfer = SectorCount - CurrentSector;
		}

		if (SectorsToTransfer == 0) break;

		Status = FileEntry->WriteSectors(FileEntry, CurrentSector + SectorStart, SectorsToTransfer, Buffer);
		if (ARC_FAIL(Status)) return Status;

		ULONG Limit = SectorsToTransfer * SectorSize;
		*Count += Limit;
		Length -= Limit;
		Buffer = (PVOID)((size_t)Buffer + Limit);
		BytesToTransfer -= Limit;
		FileEntry->Position += Limit;
		CurrentSector += SectorsToTransfer;
	}

	// If there's any data left to write, read the last sector seperately, replace the data, and write back to disk.
	if (Length != 0) {
		Status = FileEntry->ReadSectors(FileEntry, CurrentSector + SectorStart, 1, LocalPointer);
		if (ARC_FAIL(Status)) {
			return Status;
		}

		memcpy(LocalPointer, Buffer, Length);

		Status = FileEntry->WriteSectors(FileEntry, CurrentSector + SectorStart, 1, LocalPointer);
		if (ARC_FAIL(Status)) return Status;

		*Count += Length;
		FileEntry->Position += Length;
	}

	return _ESUCCESS;
}

static ARC_STATUS UsbDiskWrite(PARC_FILE_TABLE FileEntry, ULONG StartSector, ULONG CountSectors, PVOID Buffer) {
	PUSB_DEVICE_MOUNT_ENTRY MountEntry = FileEntry->u.DiskContext.DeviceMount;
	if (MountEntry == NULL) return _EBADF;

	ULONG WrittenSectors = UlmsWriteSectors(MountEntry->Controller, 0, StartSector, CountSectors, Buffer);
	if (WrittenSectors != CountSectors) return _EIO;
	return _ESUCCESS;
}

static ARC_STATUS UsbDiskGetReadStatus(ULONG FileId) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;

	PUSB_DEVICE_MOUNT_ENTRY MountEntry = FileEntry->u.DiskContext.DeviceMount;
	if (MountEntry == NULL) return _EBADF;

	int64_t LastByte = FileEntry->u.DiskContext.SectorCount;
	LastByte *= MountEntry->SectorSize;
	if (FileEntry->Position >= LastByte) return _EAGAIN;
	return _ESUCCESS;
}

static ARC_STATUS UsbDiskGetFileInformation(ULONG FileId, PFILE_INFORMATION FileInfo) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;
	PUSB_DEVICE_MOUNT_ENTRY MountEntry = FileEntry->u.DiskContext.DeviceMount;

	FileInfo->CurrentPosition.QuadPart = FileEntry->Position;
	int64_t Temp64 = FileEntry->u.DiskContext.SectorStart;
	Temp64 *= MountEntry->SectorSize;
	FileInfo->StartingAddress.QuadPart = Temp64;
	Temp64 = FileEntry->u.DiskContext.SectorCount;
	Temp64 *= MountEntry->SectorSize;
	FileInfo->EndingAddress.QuadPart = Temp64;
	FileInfo->Type = DiskPeripheral;

	return _ESUCCESS;
}

// Disk images on EXI/SD device vectors.
static const DEVICE_VECTORS ImgVectors = {
	.Open = ImgOpen,
	.Close = ImgClose,
	.Mount = ImgMount,
	.Read = DeblockerRead,
	.Write = DeblockerWrite,
	.Seek = DeblockerSeek,
	.GetReadStatus = ImgGetReadStatus,
	.GetFileInformation = ImgGetFileInformation,
	.SetFileInformation = NULL,
	.GetDirectoryEntry = NULL
};

ARC_STATUS ImgFfsToArc(FRESULT result) {
	switch (result) {
	case FR_OK:
		return _ESUCCESS;
	case FR_DISK_ERR:
		return _EIO;
	case FR_NOT_READY:
		return _ENXIO;
	case FR_NO_FILE:
		return _ENOENT;
	case FR_INVALID_OBJECT:
		return _EINVAL;
	case FR_NOT_ENABLED:
		return _ENXIO;
	case FR_NO_FILESYSTEM:
		return _EBADF;
	default:
		return _EFAULT;
	}
}

// IDE device functions.

static inline ARC_FORCEINLINE ULONG ImgGetSectorSizeImpl(PARC_FILE_TABLE FileEntry) {
	return FileEntry->Flags.CdImage ? 2048 : 0x200;
}

static ARC_STATUS ImgGetSectorSize(ULONG FileId, PULONG SectorSize) {
	// Get the file table entry.
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EFAULT;

	*SectorSize = ImgGetSectorSizeImpl(FileEntry);
	return _ESUCCESS;
}

enum {
	FATFS_FAST_SEEK_COUNT = 64
};

typedef struct _IMG_LINK_MAP_ENTRY IMG_LINK_MAP_ENTRY, *PIMG_LINK_MAP_ENTRY;
struct _IMG_LINK_MAP_ENTRY {
	PIMG_LINK_MAP_ENTRY Next;
	UCHAR IsValid; // 0 == unknown, 1 == valid, 2 == failed
	UCHAR DeviceIndex;
	UCHAR DiskIndex;
	ULONG LinkMap[FATFS_FAST_SEEK_COUNT];
};

static PIMG_LINK_MAP_ENTRY s_LinkMap = NULL;

static PIMG_LINK_MAP_ENTRY ImgpGetLinkMap(UCHAR DeviceIndex, UCHAR DiskIndex) {
	PIMG_LINK_MAP_ENTRY* EntryLink = &s_LinkMap;
	for (PIMG_LINK_MAP_ENTRY Entry = *EntryLink; Entry != NULL; EntryLink = &Entry->Next, Entry = *EntryLink) {
		if (Entry->DeviceIndex == DeviceIndex && Entry->DiskIndex == DiskIndex) return Entry;
	}

	// no entry present for this disk, make one
	PIMG_LINK_MAP_ENTRY Entry = malloc(sizeof(**EntryLink));
	Entry->Next = NULL;
	Entry->DeviceIndex = DeviceIndex;
	Entry->DiskIndex = DiskIndex;
	Entry->IsValid = 0;
	Entry->LinkMap[0] = FATFS_FAST_SEEK_COUNT;
	*EntryLink = Entry;
	return Entry;
}

void ImgInvalidateLinkMap(UCHAR DeviceIndex, UCHAR DiskIndex) {
	PIMG_LINK_MAP_ENTRY* EntryLink = &s_LinkMap;
	for (PIMG_LINK_MAP_ENTRY Entry = *EntryLink; Entry != NULL; EntryLink = &Entry->Next, Entry = *EntryLink) {
		if (Entry->DeviceIndex == DeviceIndex && Entry->DiskIndex == DiskIndex) {
			Entry->LinkMap[0] = FATFS_FAST_SEEK_COUNT;
			Entry->IsValid = 0;
		}
	}
}

static ARC_STATUS ImgOpen(PCHAR OpenPath, OPEN_MODE OpenMode, PULONG FileId) {
	// Ensure the path starts with s_IossdmcControllerPath or s_ExiControllerPath
	bool IsIosSdmc = false;
	ULONG DeviceIndex = 0;
	PCHAR DevicePath;

	if (memcmp(OpenPath, s_IossdmcControllerPath, sizeof(s_IossdmcControllerPath) - 1) == 0) {
		DeviceIndex = 4; // FATFS drive index for IOSSDMC controller
		DevicePath = &OpenPath[sizeof(s_IossdmcControllerPath) - 1];
		IsIosSdmc = true;
	}
	else {
		if (memcmp(OpenPath, s_ExiControllerPath, sizeof(s_ExiControllerPath) - 1) != 0) return _ENODEV;
		DevicePath = &OpenPath[sizeof(s_ExiControllerPath) - 1];
	}
	// Next device must be disk(x) or cdrom(x)
	ULONG Index = 0;
	bool IsCdRom = false;
	bool IsFloppy = false;
	if (!ArcDeviceParse(&DevicePath, DiskController, &Index)) {
		// Not disk(x), check for cdrom(x)
		IsCdRom = true;
		if (!ArcDeviceParse(&DevicePath, CdromController, &Index)) {
			// Not cdrom either, can't handle this device path
			return _ENODEV;
		}
	}

	// Got the index, make sure this is good
	if (IsIosSdmc) {
		if (Index != 0) return _ENODEV;
	}
	else {
		if (Index >= 4) return _ENODEV;
		DeviceIndex = Index;
	}

	if ((disk_status(DeviceIndex) & STA_NOINIT) != 0) return _ENODEV;


	// For cdrom, next device must be fdisk(x)
	// For disk, next device must be rdisk(x)
	// For floppy, must be disk and next device must be fdisk(0)
	ULONG ImgIdx = 0;
	if (IsCdRom) {
		if (!ArcDeviceParse(&DevicePath, FloppyDiskPeripheral, &ImgIdx)) {
			return _ENODEV;
		}
		// should be the last device, but partition is still technically valid here.
	}
	else {
		if (!ArcDeviceParse(&DevicePath, DiskPeripheral, &ImgIdx)) {
			ULONG MustBeZero = 0;
			if (!ArcDeviceParse(&DevicePath, FloppyDiskPeripheral, &MustBeZero)) {
				return _ENODEV;
			}
			IsFloppy = true;
			if (MustBeZero != 0) return _ENODEV;
		}
	}

	// Can only handle up to 100 raw images, and even that's kinda overkill ;)
	if (!IsFloppy && ImgIdx >= 100) return _ENODEV;


	// If this is a cdrom, it can only be mounted ro.
	if (IsCdRom && OpenMode != ArcOpenReadOnly) return _EACCES;

	bool IncludesPartition = false;
	ULONG PartitionNumber = 0;
	// Does the caller want a partition?
	if (*DevicePath != 0) {
		if (!ArcDeviceParse(&DevicePath, PartitionEntry, &PartitionNumber)) {
			return _ENODEV;
		}
		else {
			// partition 0 means whole disk, and is the only valid partition for cdrom or floppy
			if ((IsCdRom || IsFloppy) && PartitionNumber != 0) return _ENODEV;
			IncludesPartition = PartitionNumber != 0;
		}
	}

	// Get the file table entry.
	PARC_FILE_TABLE FileEntry = ArcIoGetFileForOpen(*FileId);
	if (FileEntry == NULL) return _EFAULT;
	// Zero the disk context.
	memset(&FileEntry->u.DiskContext, 0, sizeof(FileEntry->u.DiskContext));

	// It's now known if this is a disk or cdrom (ie, whether to use FAT or ISO9660 filesystem driver)
	// Attempt to open the raw image.
	char Path[260];
	if (IsFloppy) snprintf(Path, sizeof(Path), "%d:/nt/drivers.img", DeviceIndex);
	else snprintf(Path, sizeof(Path), "%d:/nt/disk%02d.%s", DeviceIndex, ImgIdx, IsCdRom ? "iso" : "img");
	// Make sure the file exists.
	FILINFO info;
	if (f_stat(Path, &info) != FR_OK) return _ENODEV;

	// Open the file.
	FIL* pFile = malloc(sizeof(FIL) + ((sizeof(ULONG) * FATFS_FAST_SEEK_COUNT)));
	if (pFile == NULL) return _ENOMEM;
	if (f_open(pFile, Path, FA_READ | FA_WRITE) != FR_OK) {
		free(pFile);
		return _ENODEV;
	}

	// Set up the link map.
	USHORT LinkIndex = ImgIdx;
	if (IsCdRom) LinkIndex += 100;
	if (IsFloppy) LinkIndex = 200;
	PIMG_LINK_MAP_ENTRY Entry = ImgpGetLinkMap(DeviceIndex, LinkIndex);
	if (Entry != NULL && Entry->IsValid < 2) {
		pFile->cltbl = Entry->LinkMap;
		if (!Entry->IsValid) {
			if (f_lseek(pFile, CREATE_LINKMAP) != FR_OK) {
				pFile->cltbl = NULL;
				Entry->IsValid = 2;
			}
			else {
				Entry->IsValid = 1;
			}
		}
	}

	FileEntry->u.DiskContext.FatfsFile = pFile;
	FileEntry->Flags.CdImage = IsCdRom;

	ULONG SectorSize = ImgGetSectorSizeImpl(FileEntry);

	// Stash the GetSectorSize ptr into the file table.
	FileEntry->GetSectorSize = ImgGetSectorSize;
	FileEntry->ReadSectors = ImgRead;
	FileEntry->WriteSectors = ImgWrite;
	ULONG MaxSectorTransfer = 1;
	if (disk_ioctl(DeviceIndex, 100, &MaxSectorTransfer) != RES_OK) MaxSectorTransfer = 1;
	if (IsCdRom) {
		if (MaxSectorTransfer <= (0x800 / 0x200)) MaxSectorTransfer = 1;
		else MaxSectorTransfer /= (0x800 / 0x200);
	}
	FileEntry->u.DiskContext.MaxSectorTransfer = MaxSectorTransfer;

	ULONG DiskSectors;
	if (SectorSize == 0x200) DiskSectors = info.fsize / 0x200;
	else if (SectorSize == 0x800) DiskSectors = info.fsize / 0x800;
	else DiskSectors = info.fsize / SectorSize;
	ULONG PartitionSectors = DiskSectors;
	ULONG PartitionSector = 0;
	// Set it up for the whole disk so ArcFsPartitionObtain can work.
	FileEntry->u.DiskContext.SectorStart = PartitionSector;
	FileEntry->u.DiskContext.SectorCount = PartitionSectors;

	ARC_STATUS Status;
	if (IncludesPartition) {
		// Mark the file as open so ArcFsPartitionObtain can work.
		FileEntry->Flags.Open = 1;
		Status = ArcFsPartitionObtain(FileEntry->DeviceEntryTable, *FileId, PartitionNumber, SectorSize, &PartitionSector, &PartitionSectors);
		FileEntry->Flags.Open = 0;
		if (ARC_FAIL(Status)) {
#if 0
			// osloader uses partition 1 if it doesn't see any, so if the caller asks for that and it wasn't found, give them the whole disk
			if (PartitionNumber != 1) {
				fclose(f);
				return _ENODEV;
			}
#endif
			PartitionSector = 0;
			PartitionSectors = DiskSectors;
		}
	}
	FileEntry->u.DiskContext.SectorStart = PartitionSector;
	FileEntry->u.DiskContext.SectorCount = PartitionSectors;

	FileEntry->Position = 0;

	return _ESUCCESS;
}

static ARC_STATUS ImgClose(ULONG FileId) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;
	ARC_STATUS status = ImgFfsToArc(f_close(FileEntry->u.DiskContext.FatfsFile));
	if (ARC_FAIL(status)) return status;
	free(FileEntry->u.DiskContext.FatfsFile);
	return status;
}

static ARC_STATUS ImgMount(PCHAR MountPath, MOUNT_OPERATION Operation) { return _EINVAL; }

static ARC_STATUS ImgRead(PARC_FILE_TABLE FileEntry, ULONG StartSector, ULONG CountSectors, PVOID Buffer) {
	UINT NumberOfBytesRead;
	FSIZE_t offset = StartSector;
	UINT CountBytes = CountSectors;
	ULONG SectorSize = ImgGetSectorSizeImpl(FileEntry);
	// Hardcode the two most common sector sizes to avoid expensive divdi3 calls
	if (SectorSize == 0x200) {
		offset *= 0x200;
		CountBytes *= 0x200;
	}
	else if (SectorSize == 0x800) {
		offset *= 0x800;
		CountBytes *= 0x800;
	}
	else {
		offset *= SectorSize;
		CountBytes *= SectorSize;
	}

	FIL* file = FileEntry->u.DiskContext.FatfsFile;

	FRESULT result = f_lseek(file, offset);
	if (result != FR_OK) return ImgFfsToArc(result);
	result = f_read(file, Buffer, CountBytes, &NumberOfBytesRead);
	if (result != FR_OK) return ImgFfsToArc(result);
	bool Success = NumberOfBytesRead == CountBytes;
	if (!Success) return _EIO;
	return _ESUCCESS;
}

static ARC_STATUS ImgWrite(PARC_FILE_TABLE FileEntry, ULONG StartSector, ULONG CountSectors, PVOID Buffer) {
	if (FileEntry->Flags.CdImage) return _EROFS;
	UINT NumberOfBytesWrote;
	FSIZE_t offset = StartSector;
	UINT CountBytes = CountSectors;
	ULONG SectorSize = ImgGetSectorSizeImpl(FileEntry);
	// Hardcode the two most common sector sizes to avoid expensive divdi3 calls
	if (SectorSize == 0x200) {
		offset *= 0x200;
		CountBytes *= 0x200;
	}
	else if (SectorSize == 0x800) {
		offset *= 0x800;
		CountBytes *= 0x800;
	}
	else {
		offset *= SectorSize;
		CountBytes *= SectorSize;
	}

	FIL* file = FileEntry->u.DiskContext.FatfsFile;

	FRESULT result = f_lseek(file, offset);
	if (result != FR_OK) return ImgFfsToArc(result);
	result = f_write(file, Buffer, CountBytes, &NumberOfBytesWrote);
	if (result != FR_OK) return ImgFfsToArc(result);
	bool Success = NumberOfBytesWrote == CountBytes;
	if (!Success) return _EIO;
	return _ESUCCESS;
}

static ARC_STATUS ImgGetReadStatus(ULONG FileId) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;
	int64_t LastByte = FileEntry->u.DiskContext.SectorCount;
	LastByte *= ImgGetSectorSizeImpl(FileEntry);
	if (FileEntry->Position >= LastByte) return _EAGAIN;
	return _ESUCCESS;
}

static ARC_STATUS ImgGetFileInformation(ULONG FileId, PFILE_INFORMATION FileInfo) {
	PARC_FILE_TABLE FileEntry = ArcIoGetFile(FileId);
	if (FileEntry == NULL) return _EBADF;
	ULONG SectorSize = ImgGetSectorSizeImpl(FileEntry);

	FileInfo->CurrentPosition.QuadPart = FileEntry->Position;
	int64_t Temp64 = FileEntry->u.DiskContext.SectorStart;
	Temp64 *= SectorSize;
	FileInfo->StartingAddress.QuadPart = Temp64;
	Temp64 = FileEntry->u.DiskContext.SectorCount;
	Temp64 *= SectorSize;
	FileInfo->EndingAddress.QuadPart = Temp64;
	FileInfo->Type = DiskPeripheral;

	return _ESUCCESS;
}

static bool ArcDiskUsbInit() {
	// Get the disk component.
	PCONFIGURATION_COMPONENT DiskComponent = ARC_VENDOR_VECTORS()->GetComponentRoutine(s_UsbControllerPath);
	if (DiskComponent == NULL) return false;
	// Ensure that the component obtained was really the disk component.
	if (DiskComponent->Class != AdapterClass) return false;
	if (DiskComponent->Type != ScsiAdapter) return false;
	if (DiskComponent->Key != 0) return false;

	// We really have a pointer to a device entry.
	PDEVICE_ENTRY DiskDevice = (PDEVICE_ENTRY)DiskComponent;

	DiskDevice->Vectors = &UsbDiskVectors;
	DiskDevice->Component.Identifier = (size_t)s_UsbComponentName;
	DiskDevice->Component.IdentifierLength = sizeof(s_UsbComponentName);
	return true;
}

static bool ArcDiskImgInit() {
	// Get the disk component.
	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
	PCONFIGURATION_COMPONENT ExiComponent = Api->GetComponentRoutine(s_ExiControllerPath);
	PCONFIGURATION_COMPONENT IossdmcComponent = Api->GetComponentRoutine(s_IossdmcControllerPath);
	// Ensure that the component obtained was really the disk component.
	if (ExiComponent->Class != AdapterClass) return false;
	if (ExiComponent->Type != MultiFunctionAdapter) return false;
	if (ExiComponent->Key != 0) return false;
	if (IossdmcComponent != NULL) {
		if (IossdmcComponent->Class != AdapterClass) return false;
		if (IossdmcComponent->Type != MultiFunctionAdapter) return false;
		if (IossdmcComponent->Key != 1) return false;
	}

	// We really have a pointer to a device entry.
	PDEVICE_ENTRY ExiDevice = (PDEVICE_ENTRY)ExiComponent;
	PDEVICE_ENTRY IossdmcDevice = (PDEVICE_ENTRY)IossdmcComponent;

	ExiDevice->Vectors = &ImgVectors;

	for (int i = 0; i < 4; i++) {
		if ((disk_status(i) & STA_NOINIT) != 0) continue;

		// Add the disk controller
		CONFIGURATION_COMPONENT ControllerConfig = ARC_MAKE_COMPONENT(ControllerClass, DiskController, ARC_DEVICE_INPUT | ARC_DEVICE_OUTPUT, i, 0);
		PDEVICE_ENTRY ControllerDevice = (PDEVICE_ENTRY)Api->AddChildRoutine(&ExiDevice->Component, &ControllerConfig, NULL);
		if (ControllerDevice != NULL) {
			ControllerDevice->Vectors = ExiDevice->Vectors;
		}

		// Add the cdrom controller
		ControllerConfig = ARC_MAKE_COMPONENT(ControllerClass, CdromController, ARC_DEVICE_INPUT | ARC_DEVICE_OUTPUT, i, 0);
		ControllerDevice = (PDEVICE_ENTRY)Api->AddChildRoutine(&ExiDevice->Component, &ControllerConfig, NULL);
		if (ControllerDevice != NULL) {
			ControllerDevice->Vectors = ExiDevice->Vectors;
		}
	}

	if (IossdmcDevice != NULL) {
		IossdmcDevice->Vectors = &ImgVectors;

		IossdmcDevice->Component.Identifier = (size_t)s_IossdmcComponentName;
		IossdmcDevice->Component.IdentifierLength = sizeof(s_IossdmcComponentName);

		// Add the disk controller
		CONFIGURATION_COMPONENT ControllerConfig = ARC_MAKE_COMPONENT(ControllerClass, DiskController, ARC_DEVICE_INPUT | ARC_DEVICE_OUTPUT, 0, 0);
		PDEVICE_ENTRY ControllerDevice = (PDEVICE_ENTRY)Api->AddChildRoutine(&IossdmcDevice->Component, &ControllerConfig, NULL);
		if (ControllerDevice != NULL) {
			ControllerDevice->Vectors = IossdmcDevice->Vectors;
		}

		// Add the cdrom controller
		ControllerConfig = ARC_MAKE_COMPONENT(ControllerClass, CdromController, ARC_DEVICE_INPUT | ARC_DEVICE_OUTPUT, 0, 0);
		ControllerDevice = (PDEVICE_ENTRY)Api->AddChildRoutine(&IossdmcDevice->Component, &ControllerConfig, NULL);
		if (ControllerDevice != NULL) {
			ControllerDevice->Vectors = IossdmcDevice->Vectors;
		}
	}
	return true;
}

static bool ArcDiskAddDevice(PVENDOR_VECTOR_TABLE Api, PDEVICE_ENTRY BaseController, CONFIGURATION_TYPE Controller, CONFIGURATION_TYPE Disk, ULONG Key) {
	CONFIGURATION_COMPONENT ControllerConfig = ARC_MAKE_COMPONENT(ControllerClass, Controller, ARC_DEVICE_INPUT | ARC_DEVICE_OUTPUT, Key, 0);
	PDEVICE_ENTRY ControllerDevice = (PDEVICE_ENTRY)Api->AddChildRoutine(&BaseController->Component, &ControllerConfig, NULL);
	if (ControllerDevice == NULL) return false; // can't do anything if AddChild did fail
	ControllerDevice->Vectors = BaseController->Vectors;

	CONFIGURATION_COMPONENT DiskConfig = ARC_MAKE_COMPONENT(PeripheralClass, Disk, ARC_DEVICE_INPUT | ARC_DEVICE_OUTPUT, 0, 0);
	PDEVICE_ENTRY DiskDevice = (PDEVICE_ENTRY)Api->AddChildRoutine(&ControllerDevice->Component, &DiskConfig, NULL);
	if (DiskDevice == NULL) return false; // can't do anything if AddChild did fail
	DiskDevice->Vectors = BaseController->Vectors;
	return true;
}

static bool ArcDiskAddDeviceImg(PVENDOR_VECTOR_TABLE Api, PDEVICE_ENTRY BaseController, CONFIGURATION_TYPE Disk, ULONG Key) {
	if (BaseController == NULL) return false;
	CONFIGURATION_COMPONENT DiskConfig = ARC_MAKE_COMPONENT(PeripheralClass, Disk, ARC_DEVICE_INPUT | ARC_DEVICE_OUTPUT, Key, 0);
	PDEVICE_ENTRY DiskDevice = (PDEVICE_ENTRY)Api->AddChildRoutine(&BaseController->Component, &DiskConfig, NULL);
	if (DiskDevice == NULL) return false; // can't do anything if AddChild did fail
	DiskDevice->Vectors = BaseController->Vectors;
	return true;
}

static ULONG s_CountCdrom = 0;
static ULONG s_CountDisk = 0;
static ULONG s_CountPartitions[100] = { 0 };
static ULONG s_SizeDiskMb[100] = { 0 };

void ArcDiskGetCounts(PULONG Disk, PULONG Cdrom) {
	if (Disk != NULL) *Disk = s_CountDisk;
	if (Cdrom != NULL) *Cdrom = s_CountCdrom;
}

ULONG ArcDiskGetPartitionCount(ULONG Disk) {
	if (Disk >= s_CountDisk) return 0;
	return s_CountPartitions[Disk];
}

ULONG ArcDiskGetSizeMb(ULONG Disk) {
	if (Disk >= s_CountDisk) return 0;
	return s_SizeDiskMb[Disk];
}

void ArcDiskInit() {
	ArcDiskImgInit();
	ArcDiskUsbInit();

	printf("Scanning disk devices...\r\n");

	char EnvKeyCd[] = "cd00:";
	char EnvKeyHd[] = "hd00p0:";
	char EnvKeyHdRaw[] = "hd00:";
	char DeviceName[64];

	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
	PDEVICE_ENTRY UsbController = (PDEVICE_ENTRY) Api->GetComponentRoutine(s_UsbControllerPath);
	PDEVICE_ENTRY ExiControllerBase = (PDEVICE_ENTRY)Api->GetComponentRoutine(s_ExiControllerPath);
	PDEVICE_ENTRY IossdmcControllerBase = (PDEVICE_ENTRY)Api->GetComponentRoutine(s_IossdmcControllerPath);
	PDEVICE_ENTRY IossdmcControllers[2] = { NULL, NULL };
	PDEVICE_ENTRY ExiControllers[2] = { NULL, NULL };
	if (IossdmcControllerBase != NULL) {
		snprintf(DeviceName, sizeof(DeviceName), "%sdisk(0)", s_IossdmcControllerPath);
		IossdmcControllers[0] = (PDEVICE_ENTRY)Api->GetComponentRoutine(DeviceName);
		snprintf(DeviceName, sizeof(DeviceName), "%scdrom(0)", s_IossdmcControllerPath);
		IossdmcControllers[1] = (PDEVICE_ENTRY)Api->GetComponentRoutine(DeviceName);
	}

	ULONG CdromCount = 0;
	ULONG DiskCount = 0;

	// USB mass storage devices first.
	USBMS_DEVICES UsbDevices;
	UlmsGetDevices(&UsbDevices);

	for (ULONG i = 0; UsbController != NULL && i < UsbDevices.DeviceCount; i++) {
		// Get the sector size.
		ULONG Key = UsbDevices.ArcKey[i];
		PUSBMS_CONTROLLER Controller = UlmsGetController(Key);
		if (Controller == NULL) continue;
		if (UlmsGetLuns(Controller) == 0) continue;

		ULONG SectorSize = UlmsGetSectorSize(Controller, 0);
		if (SectorSize == 2048) {
			// This looks like an optical drive.
			if (CdromCount < 100) {
				// Add the cdrom device.
				if (!ArcDiskAddDevice(Api, UsbController, CdromController, FloppyDiskPeripheral, Key)) continue;

				EnvKeyCd[3] = (CdromCount % 10) + '0';
				EnvKeyCd[2] = ((CdromCount / 10) % 10) + '0';
				CdromCount++;
				snprintf(DeviceName, sizeof(DeviceName), "%scdrom(%u)fdisk(0)", s_UsbControllerPath, Key);
				ArcEnvSetDevice(EnvKeyCd, DeviceName);
				printf("%s - USB device VID=%04x,PID=%04x\r\n", EnvKeyCd, (Key >> 16) & 0xFFFF, Key & 0xFFFF);
			}
		}
		else if (SectorSize == 0x200) {
			// This looks like a hard drive.
			if (DiskCount < 100) {
				// Add the raw drive device.
				if (!ArcDiskAddDevice(Api, UsbController, DiskController, DiskPeripheral, Key)) continue;
				EnvKeyHdRaw[3] = (DiskCount % 10) + '0';
				EnvKeyHdRaw[2] = ((DiskCount / 10) % 10) + '0';
				EnvKeyHd[3] = EnvKeyHdRaw[3];
				EnvKeyHd[2] = EnvKeyHdRaw[2];
				ULONG DiskIndex = DiskCount;
				DiskCount++;
				snprintf(DeviceName, sizeof(DeviceName), "%sdisk(%u)rdisk(0)", s_UsbControllerPath, Key);
				ArcEnvSetDevice(EnvKeyHdRaw, DeviceName);
				printf("%s - USB device VID=%04x,PID=%04x (raw disk)\r\n", EnvKeyHdRaw, (Key >> 16) & 0xFFFF, Key & 0xFFFF);

				U32LE DeviceId;
				if (ARC_FAIL(Api->OpenRoutine(DeviceName, ArcOpenReadWrite, &DeviceId))) continue;
				// Attempt to get the number of MBR partitions.
				if (ARC_FAIL(ArcFsPartitionCount(DeviceId.v, &s_CountPartitions[DiskIndex]))) s_CountPartitions[DiskIndex] = 0;
				// And the size of the disk.
				{
					FILE_INFORMATION Info;
					if (ARC_SUCCESS(Api->GetFileInformationRoutine(DeviceId.v, &Info))) s_SizeDiskMb[DiskIndex] = (ULONG)(Info.EndingAddress.QuadPart / 0x100000);
				}
				Api->CloseRoutine(DeviceId.v);

				for (ULONG part = 1; part <= s_CountPartitions[DiskIndex]; part++) {
					snprintf(DeviceName, sizeof(DeviceName), "%sdisk(%u)rdisk(0)partition(%d)", s_UsbControllerPath, Key, part);
					if (part < 10) {
						EnvKeyHd[5] = part + '0';
						EnvKeyHd[6] = ':';
						EnvKeyHd[7] = 0;
					}
					else {
						EnvKeyHd[6] = (part % 10) + '0';
						EnvKeyHd[5] = ((part / 10) % 10) + '0';
						EnvKeyHd[7] = ':';
					}
					ArcEnvSetDevice(EnvKeyHd, DeviceName);
					printf("%s - USB device VID=%04x,PID=%04x (partition %d)\r\n", EnvKeyHd, (Key >> 16) & 0xFFFF, Key & 0xFFFF, part);
				}
			}
		}
	}

	// Now IOSSDMC
	if ((disk_status(4) & STA_NOINIT) == 0 && IossdmcControllers[0] != NULL && IossdmcControllers[1] != NULL) {
		// Check the floppy image first.
		U32LE DeviceId;
		snprintf(DeviceName, sizeof(DeviceName), "%sdisk(0)fdisk(0)", s_IossdmcControllerPath);
		if (ARC_SUCCESS(Api->OpenRoutine(DeviceName, ArcOpenReadOnly, &DeviceId))) {
			Api->CloseRoutine(DeviceId.v);
			if (ArcDiskAddDeviceImg(Api, IossdmcControllers[0], FloppyDiskPeripheral, 0)) {
				printf("SD drivers.img present\r\n");
			}
		}

		bool CheckHd = true;
		bool CheckCd = true;

		for (BYTE i = 0; i < 100 && (CheckHd || CheckCd); i++) {
			do {
				if (CheckCd && CdromCount >= 100) CheckCd = false;
				if (CheckCd) {
					snprintf(DeviceName, sizeof(DeviceName), "%scdrom(0)fdisk(%d)", s_IossdmcControllerPath, i);
					if (ARC_FAIL(Api->OpenRoutine(DeviceName, ArcOpenReadOnly, &DeviceId))) {
						CheckCd = false;
						break;
					}

					Api->CloseRoutine(DeviceId.v);

					if (!ArcDiskAddDeviceImg(Api, IossdmcControllers[1], FloppyDiskPeripheral, i)) continue;

					EnvKeyCd[3] = (CdromCount % 10) + '0';
					EnvKeyCd[2] = ((CdromCount / 10) % 10) + '0';
					CdromCount++;
					ArcEnvSetDevice(EnvKeyCd, DeviceName);
					printf("%s - SD image %d\r\n", EnvKeyCd, i);
				}
			} while (false);

			do {
				if (CheckHd && DiskCount >= 100) CheckHd = false;
				if (CheckHd) {
					snprintf(DeviceName, sizeof(DeviceName), "%sdisk(0)rdisk(%d)", s_IossdmcControllerPath, i);
					if (ARC_FAIL(Api->OpenRoutine(DeviceName, ArcOpenReadOnly, &DeviceId))) {
						CheckHd = false;
						break;
					}
					
					ULONG DiskIndex = DiskCount;

					// Attempt to get the number of MBR partitions.
					if (ARC_FAIL(ArcFsPartitionCount(DeviceId.v, &s_CountPartitions[DiskIndex]))) s_CountPartitions[DiskIndex] = 0;
					// And the size of the disk.
					{
						FILE_INFORMATION Info;
						if (ARC_SUCCESS(Api->GetFileInformationRoutine(DeviceId.v, &Info))) s_SizeDiskMb[DiskIndex] = (ULONG)(Info.EndingAddress.QuadPart / 0x100000);
					}

					Api->CloseRoutine(DeviceId.v);

					// Set up friendly names for any existing partitions, and the raw drive.

					if (!ArcDiskAddDeviceImg(Api, IossdmcControllers[0], DiskPeripheral, i)) continue;
					EnvKeyHdRaw[3] = (DiskCount % 10) + '0';
					EnvKeyHdRaw[2] = ((DiskCount / 10) % 10) + '0';
					EnvKeyHd[3] = EnvKeyHdRaw[3];
					EnvKeyHd[2] = EnvKeyHdRaw[2];
					DiskCount++;
					snprintf(DeviceName, sizeof(DeviceName), "%sdisk(0)rdisk(%d)", s_IossdmcControllerPath, i);
					ArcEnvSetDevice(EnvKeyHdRaw, DeviceName);
					printf("%s - SD image %d (raw disk)\r\n", EnvKeyHdRaw, i);
					for (ULONG part = 1; part <= s_CountPartitions[DiskIndex]; part++) {
						snprintf(DeviceName, sizeof(DeviceName), "%sdisk(0)rdisk(%d)partition(%d)", s_IossdmcControllerPath, i, part);
						if (part < 10) {
							EnvKeyHd[5] = part + '0';
							EnvKeyHd[6] = ':';
							EnvKeyHd[7] = 0;
						}
						else {
							EnvKeyHd[6] = (part % 10) + '0';
							EnvKeyHd[5] = ((part / 10) % 10) + '0';
							EnvKeyHd[7] = ':';
						}
						ArcEnvSetDevice(EnvKeyHd, DeviceName);
						printf("%s - SD image %d (partition %d)\r\n", EnvKeyHd, i, part);
					}
				}
			} while (false);
		}
	}

	// Now EXI devices

	static const char s_ExiDevices[][4] = {
		{ 'M', 'C', 'A', 0 },
		{ 'M', 'C', 'B', 0 },
		{ 'S', 'P', '1', 0 },
		{ 'S', 'P', '2', 0 }
	};

	for (ULONG exi = 0; exi < 4; exi++) {
		if (ExiControllerBase == NULL) break;
		if ((disk_status(exi) & STA_NOINIT) != 0) continue;
		snprintf(DeviceName, sizeof(DeviceName), "%sdisk(%d)", s_ExiControllerPath, exi);
		ExiControllers[0] = (PDEVICE_ENTRY)Api->GetComponentRoutine(DeviceName);
		snprintf(DeviceName, sizeof(DeviceName), "%scdrom(%d)", s_ExiControllerPath, exi);
		ExiControllers[1] = (PDEVICE_ENTRY)Api->GetComponentRoutine(DeviceName);

		if (ExiControllers[0] == NULL || ExiControllers[1] == NULL) continue;

		// Check the floppy image first.
		U32LE DeviceId;
		snprintf(DeviceName, sizeof(DeviceName), "%sdisk(%d)fdisk(0)", s_ExiControllerPath, exi);
		if (ARC_SUCCESS(Api->OpenRoutine(DeviceName, ArcOpenReadOnly, &DeviceId))) {
			Api->CloseRoutine(DeviceId.v);
			if (ArcDiskAddDeviceImg(Api, ExiControllers[0], FloppyDiskPeripheral, 0)) {
				printf("%s drivers.img present\r\n", s_ExiDevices[exi]);
			}
		}

		bool CheckHd = true;
		bool CheckCd = true;

		for (BYTE i = 0; i < 100 && (CheckHd || CheckCd); i++) {
			do {
				if (CheckCd && CdromCount >= 100) CheckCd = false;
				if (CheckCd) {
					snprintf(DeviceName, sizeof(DeviceName), "%scdrom(%d)fdisk(%d)", s_ExiControllerPath, exi, i);
					if (ARC_FAIL(Api->OpenRoutine(DeviceName, ArcOpenReadOnly, &DeviceId))) {
						CheckCd = false;
						break;
					}

					Api->CloseRoutine(DeviceId.v);

					if (!ArcDiskAddDeviceImg(Api, ExiControllers[1], FloppyDiskPeripheral, i)) continue;

					EnvKeyCd[3] = (CdromCount % 10) + '0';
					EnvKeyCd[2] = ((CdromCount / 10) % 10) + '0';
					CdromCount++;
					ArcEnvSetDevice(EnvKeyCd, DeviceName);
					printf("%s - %s image %d\r\n", EnvKeyCd, s_ExiDevices[exi], i);
				}
			} while (false);

			do {
				if (CheckHd && DiskCount >= 100) CheckHd = false;
				if (CheckHd) {
					snprintf(DeviceName, sizeof(DeviceName), "%sdisk(%d)rdisk(%d)", s_ExiControllerPath, exi, i);
					if (ARC_FAIL(Api->OpenRoutine(DeviceName, ArcOpenReadOnly, &DeviceId))) {
						CheckHd = false;
						break;
					}

					ULONG DiskIndex = DiskCount;

					// Attempt to get the number of MBR partitions.
					if (ARC_FAIL(ArcFsPartitionCount(DeviceId.v, &s_CountPartitions[DiskIndex]))) s_CountPartitions[DiskIndex] = 0;
					// And the size of the disk.
					{
						FILE_INFORMATION Info;
						if (ARC_SUCCESS(Api->GetFileInformationRoutine(DeviceId.v, &Info))) s_SizeDiskMb[DiskIndex] = (ULONG)(Info.EndingAddress.QuadPart / 0x100000);
					}

					Api->CloseRoutine(DeviceId.v);

					// Set up friendly names for any existing partitions, and the raw drive.

					if (!ArcDiskAddDeviceImg(Api, ExiControllers[0], DiskPeripheral, i)) continue;
					EnvKeyHdRaw[3] = (DiskCount % 10) + '0';
					EnvKeyHdRaw[2] = ((DiskCount / 10) % 10) + '0';
					EnvKeyHd[3] = EnvKeyHdRaw[3];
					EnvKeyHd[2] = EnvKeyHdRaw[2];
					DiskCount++;
					snprintf(DeviceName, sizeof(DeviceName), "%sdisk(%d)rdisk(%d)", s_ExiControllerPath, exi, i);
					ArcEnvSetDevice(EnvKeyHdRaw, DeviceName);
					printf("%s - %s image %d (raw disk)\r\n", EnvKeyHdRaw, s_ExiDevices[exi], i);
					for (ULONG part = 1; part <= s_CountPartitions[DiskIndex]; part++) {
						snprintf(DeviceName, sizeof(DeviceName), "%sdisk(%d)rdisk(%d)partition(%d)", s_ExiControllerPath, exi, i, part);
						if (part < 10) {
							EnvKeyHd[5] = part + '0';
							EnvKeyHd[6] = ':';
							EnvKeyHd[7] = 0;
						}
						else {
							EnvKeyHd[6] = (part % 10) + '0';
							EnvKeyHd[5] = ((part / 10) % 10) + '0';
							EnvKeyHd[7] = ':';
						}
						ArcEnvSetDevice(EnvKeyHd, DeviceName);
						printf("%s - %s image %d (partition %d)\r\n", EnvKeyHd, s_ExiDevices[exi], i, part);
					}
				}
			} while (false);
		}
	}

	s_CountCdrom = CdromCount;
	s_CountDisk = DiskCount;

	printf("Complete, found %d HDs, %d optical drives\r\n", DiskCount, CdromCount);
}