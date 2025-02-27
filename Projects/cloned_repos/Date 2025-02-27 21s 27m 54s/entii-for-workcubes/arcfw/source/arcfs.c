#include <stddef.h>
#include <memory.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include "arc.h"
#include "arcdevice.h"
#include "arcconfig.h"
#include "arcdisk.h"
#include "arcenv.h"
#include "arcio.h"
#include "arcfs.h"
#include "coff.h"
#include "lib9660.h"
#include "diskio.h"
#include "pff.h"

// Partition table parsing stuff.
typedef struct ARC_LE ARC_PACKED _PARTITION_ENTRY {
	BYTE Active;
	BYTE StartChs[3];
	BYTE Type;
	BYTE EndChs[3];
	ULONG SectorStart;
	ULONG SectorCount;
} PARTITION_ENTRY, *PPARTITION_ENTRY;

// MBR (sector 0)
typedef struct ARC_LE ARC_PACKED _MBR_SECTOR {
	BYTE MbrCode[0x1B8];
	ULONG Signature;
	USHORT Reserved;
	PARTITION_ENTRY Partitions[4];
	USHORT ValidMbr;
} MBR_SECTOR, *PMBR_SECTOR;
_Static_assert(sizeof(MBR_SECTOR) == 0x200);

enum {
	MBR_VALID_SIGNATURE = 0xAA55,
	PARTITION_TYPE_FREE = 0,
	PARTITION_TYPE_EXTENDED_CHS = 5,
	PARTITION_TYPE_EXTENDED_LBA = 0xF
};

// Boyer-Moore Horspool algorithm adapted from http://www-igm.univ-mlv.fr/~lecroq/string/node18.html#SECTION00180
static PBYTE mem_mem(PBYTE startPos, const void* pattern, size_t size, size_t patternSize)
{
	const BYTE* patternc = (const BYTE*)pattern;
	size_t table[256];

	// Preprocessing
	for (ULONG i = 0; i < 256; i++)
		table[i] = patternSize;
	for (size_t i = 0; i < patternSize - 1; i++)
		table[patternc[i]] = patternSize - i - 1;

	// Searching
	size_t j = 0;
	while (j <= size - patternSize)
	{
		BYTE c = startPos[j + patternSize - 1];
		if (patternc[patternSize - 1] == c && memcmp(pattern, startPos + j, patternSize - 1) == 0)
			return startPos + j;
		j += table[c];
	}

	return NULL;
}

/// <summary>
/// Gets the start and length of a partition.
/// </summary>
/// <param name="DeviceVectors">Device function table.</param>
/// <param name="DeviceId">Device ID.</param>
/// <param name="PartitionId">Partition number to obtain (1-indexed)</param>
/// <param name="SectorSize">Sector size for the device.</param>
/// <param name="SectorStart">On success obtains the start sector for the partition</param>
/// <param name="SectorCount">On success obtains the number of sectors of the partition</param>
/// <returns>ARC status code.</returns>
ARC_STATUS ArcFsPartitionObtain(PDEVICE_VECTORS DeviceVectors, ULONG DeviceId, ULONG PartitionId, ULONG SectorSize, PULONG SectorStart, PULONG SectorCount) {
	ULONG CurrentPartition = 0;

	// Seek to sector zero.
	ULONG PositionSector = 0;

	do {
		// Seek to MBR or extended partition position.
		int64_t Position64 = PositionSector;
		Position64 *= SectorSize;
		LARGE_INTEGER Position = INT64_TO_LARGE_INTEGER(Position64);
		ARC_STATUS Status = DeviceVectors->Seek(DeviceId, &Position, SeekAbsolute);
		if (ARC_FAIL(Status)) return Status;

		// Read single sector.
		MBR_SECTOR Mbr;
		ULONG Count;
		Status = DeviceVectors->Read(DeviceId, &Mbr, sizeof(Mbr), &Count);
		if (ARC_FAIL(Status)) return Status;
		if (Count != sizeof(Mbr)) return _EBADF;

		// Ensure MBR looks valid.
		if (Mbr.ValidMbr != MBR_VALID_SIGNATURE) return _EBADF;

		// Walk through all partitions in the MBR
		// Save off a pointer to the extended partition, if needed.
		PPARTITION_ENTRY ExtendedPartition = NULL;
		for (BYTE i = 0; i < sizeof(Mbr.Partitions) / sizeof(Mbr.Partitions[0]); i++) {
			BYTE Type = Mbr.Partitions[i].Type;
			if (Type == PARTITION_TYPE_EXTENDED_CHS || Type == PARTITION_TYPE_EXTENDED_LBA) {
				if (ExtendedPartition == NULL) ExtendedPartition = &Mbr.Partitions[i];
				else {
					// More than one extended partition is invalid.
					// No operation, for now, just only use the first extended partition.
				}
				continue;
			}
			if (Type == PARTITION_TYPE_FREE) continue;
			CurrentPartition++;
			if (CurrentPartition == PartitionId) {
				// Found the wanted partition.
				*SectorStart = PositionSector + Mbr.Partitions[i].SectorStart;
				*SectorCount = Mbr.Partitions[i].SectorCount;
				return _ESUCCESS;
			}
		}

		// Didn't find the partition.
		// If there's no extended partition, then error.
		if (ExtendedPartition == NULL) return _ENODEV;
		if (ExtendedPartition->SectorCount == 0) return _ENODEV;

		// Seek to extended partition.
		PositionSector += ExtendedPartition->SectorStart;
	} while (true);
	// should never reach here
	return _ENODEV;
}

/// <summary>
/// Gets the number of partitions on a disk (MBR only).
/// </summary>
/// <param name="DeviceId">Device ID.</param>
/// <param name="PartitionCount">On success obtains the number of partitions</param>
/// <returns>ARC status code</returns>
ARC_STATUS ArcFsPartitionCount(ULONG DeviceId, PULONG PartitionCount) {
	if (PartitionCount == NULL) return _EINVAL;
	// Get the device.
	PARC_FILE_TABLE Device = ArcIoGetFile(DeviceId);
	if (Device == NULL) return _EBADF;
	// Can't be a file.
	if (Device->DeviceId != FILE_IS_RAW_DEVICE) return _EBADF;
	if (Device->GetSectorSize == NULL) return _EBADF;
	ULONG SectorSize;
	if (ARC_FAIL(Device->GetSectorSize(DeviceId, &SectorSize))) return _EBADF;
	// Must have a sector size at least 512 bytes.
	if (SectorSize < sizeof(MBR_SECTOR)) return _EBADF;
	PDEVICE_VECTORS DeviceVectors = Device->DeviceEntryTable;

	//*PartitionCount = ArcFsApmPartitionCount(DeviceVectors, DeviceId, Device->u.DiskContext.SectorSize);
	*PartitionCount = ArcFsMbrPartitionCount(DeviceVectors, DeviceId, SectorSize);
	return _ESUCCESS;
}

/// <summary>
/// Gets the number of MBR partitions on a disk.
/// </summary>
/// <param name="DeviceVectors">Device function table.</param>
/// <param name="DeviceId">Device ID.</param>
/// <param name="SectorSize">Sector size for the device.</param>
/// <returns>Number of partitions or 0 on failure</returns>
ULONG ArcFsMbrPartitionCount(PDEVICE_VECTORS DeviceVectors, ULONG DeviceId, ULONG SectorSize) {
	ULONG MbrPartitionCount = 0;

	// Seek to sector zero.
	ULONG PositionSector = 0;

	do {
		// Seek to MBR or extended partition position.
		int64_t Position64 = PositionSector;
		Position64 *= SectorSize;
		LARGE_INTEGER Position = INT64_TO_LARGE_INTEGER(Position64);
		ARC_STATUS Status = DeviceVectors->Seek(DeviceId, &Position, SeekAbsolute);
		if (ARC_FAIL(Status)) break;

		// Read single sector.
		MBR_SECTOR Mbr;
		ULONG Count;
		Status = DeviceVectors->Read(DeviceId, &Mbr, sizeof(Mbr), &Count);
		if (ARC_FAIL(Status)) break;
		if (Count != sizeof(Mbr)) break;

		// Ensure MBR looks valid.
		if (Mbr.ValidMbr != MBR_VALID_SIGNATURE) break;

		// Walk through all partitions in the MBR
		// Save off a pointer to the extended partition, if needed.
		PPARTITION_ENTRY ExtendedPartition = NULL;
		for (BYTE i = 0; i < sizeof(Mbr.Partitions) / sizeof(Mbr.Partitions[0]); i++) {
			BYTE Type = Mbr.Partitions[i].Type;
			if (Type == PARTITION_TYPE_EXTENDED_CHS || Type == PARTITION_TYPE_EXTENDED_LBA) {
				if (ExtendedPartition == NULL) ExtendedPartition = &Mbr.Partitions[i];
				else {
					// More than one extended partition is invalid.
					// No operation, for now, just only use the first extended partition.
				}
				continue;
			}
			if (Type == PARTITION_TYPE_FREE) continue;
			MbrPartitionCount++;
		}

		// Didn't find the partition.
		// If there's no extended partition, then MBR has been successfully enumerated.
		if (ExtendedPartition == NULL) break;
		if (ExtendedPartition->SectorCount == 0) break;

		// Seek to extended partition.
		PositionSector += ExtendedPartition->SectorStart;
	} while (true);
	return MbrPartitionCount;
}

static unsigned int Crc32(PVOID Buffer, ULONG Length) {
	ULONG i;
	int j;
	unsigned int byte, crc, mask;

	PUCHAR Message = (PUCHAR)Buffer;

	i = 0;
	crc = 0xFFFFFFFF;
	while (i < Length) {
		byte = Message[i];            // Get next byte.
		crc = crc ^ byte;
		for (j = 7; j >= 0; j--) {    // Do eight times.
			mask = -(crc & 1);
			crc = (crc >> 1) ^ (0xEDB88320 & mask);
		}
		i = i + 1;
	}
	return ~crc;
}

static USHORT RppUcs2UpperCaseByTable(USHORT Char) {
	static const USHORT sc_UnicodeTables[] = {
		// This table is the data from l_intl.nls, but without the 2-element header, and only the uppercase half of the data.
#include "ucs2tbl.inc"
	};

	if (Char < 'a') return Char;
	if (Char <= 'z') return (Char - 'a' + 'A');

	USHORT Offset = Char >> 8;
	Offset = sc_UnicodeTables[Offset];
	Offset += (Char >> 4) & 0xF;
	Offset = sc_UnicodeTables[Offset];
	Offset += (Char & 0xF);
	Offset = sc_UnicodeTables[Offset];

	return Char + (SHORT)Offset;
}

static ARC_STATUS RppWriteUpCaseTable(ULONG DeviceId, PDEVICE_VECTORS Vectors) {
	// Writes the $UpCase table to disk at current location.
	// We compute the table and write a sector at a time.
	BYTE Sector[0x200];
	PU16LE pLittle = (PU16LE)(ULONG)Sector;

	ULONG Char = 0;
	while (Char <= 0xFFFF) {
		// Fill in this sector
		for (ULONG i = 0; i < sizeof(Sector) / sizeof(*pLittle); i++, Char++) {
			pLittle[i].v = RppUcs2UpperCaseByTable((USHORT)Char);
		}
		// And write it to disk
		ULONG Count = 0;
		ARC_STATUS Status = Vectors->Write(DeviceId, Sector, sizeof(Sector), &Count);
		if (ARC_FAIL(Status) || Count != sizeof(Sector)) {
			if (ARC_SUCCESS(Status)) Status = _EIO;
			return Status;
		}
	}
	return _ESUCCESS;
}

static ARC_STATUS RppWriteAttrDef(ULONG DeviceId, PDEVICE_VECTORS Vectors) {
	// Writes the AttrDef file to disk at current location.
	static const BYTE sc_AttrDef[] = {
#include "attrdef.inc"
	};

	ULONG Count = 0;
	ARC_STATUS Status = Vectors->Write(DeviceId, sc_AttrDef, sizeof(sc_AttrDef), &Count);
	if (ARC_SUCCESS(Status) && Count != sizeof(sc_AttrDef)) Status = _EIO;
	if (ARC_FAIL(Status)) return Status;

	// Write the remaining data (all zero) a cluster at a time.
	BYTE Cluster[0x1000] = { 0 };
	for (ULONG i = 0; i < 0x8000 / sizeof(Cluster); i++) {
		Status = Vectors->Write(DeviceId, Cluster, sizeof(Cluster), &Count);
		if (ARC_SUCCESS(Status) && Count != sizeof(Cluster)) Status = _EIO;
		if (ARC_FAIL(Status)) return Status;
	}
	return Status;
}

static void RppDecodeRle(PBYTE Rle, ULONG Length, PBYTE Decoded, ULONG LengthOut) {
	ULONG itOut = 0;
	for (ULONG i = 0; i < Length && itOut < LengthOut; i++) {
		BYTE Value = Rle[i];
		if (Value != 0xFF) {
			Decoded[itOut] = Value;
			itOut++;
			continue;
		}
		i++;
		Value = Rle[i];
		if (Value == 0) {
			Decoded[itOut] = 0xFF;
			itOut++;
			continue;
		}

		BYTE Length = Value;
		i++;
		Value = Rle[i];
		if ((itOut + Length) > LengthOut) break;
		memset(&Decoded[itOut], Value, Length);
		itOut += Length;
	}
}

static bool RppClusterFitsIn24Bits(int64_t ClusterOffset) {
	int value = 64 - __builtin_clzll(ClusterOffset);
	return value <= 24;
}

static bool RppMftWriteCluster24(PBYTE pMft, ULONG Offset, int64_t ClusterOffset) {
	if (!RppClusterFitsIn24Bits(ClusterOffset)) return false;
	U32LE ClusterValue = { .v = (ULONG)ClusterOffset };
	BYTE LengthSize = pMft[Offset] & 0xF;
	memcpy(&pMft[Offset + LengthSize + 1], (PBYTE)(ULONG)&ClusterValue, 3);
	return true;
}

static bool RppMftWriteBadCluster24(PBYTE pMft, ULONG Offset, int64_t ClusterOffset) {
	if (!RppClusterFitsIn24Bits(ClusterOffset)) return false;
	U32LE ClusterValue = { .v = (ULONG)ClusterOffset };
	memcpy(&pMft[Offset + 1], (PBYTE)(ULONG)&ClusterValue, 3);
	return true;
}

static ARC_STATUS RpFormatNtfs(ULONG DeviceId, PDEVICE_VECTORS Vectors, ULONG StartSector, ULONG SizeMb, ULONG MbrSig) {
	// Formats a partition with size SizeMb as NTFS 1.1

	// This is the most minimialist of NTFS formatters. We hardcode boot sectors and MFTs and AttrDef and UpCase, and patch the offsets/lengths appropriately.
	// Then we write to disk boot sectors, MFT, MFTMirror, LogFile (FF-filled), AttrDef, Bitmap (make sure to calculate this correctly, with correct length!), UpCase.
	// Finally, seek to last sector and write backup boot sector.

	// BUGBUG: this is technically not 100% correct, but NT 4 autochk will recognise and fix errors anyway.

	// Given the MFT compression, this will only work for partitions up to 64GB
	// (all cluster offsets must fit into 24 bits, $BadClus has a "length" of the whole disk)
	if (SizeMb > (0x1000000 / (0x100000 / 0x1000))) {
		return _E2BIG;
	}

	static const BYTE sc_NtfsBoot[] = {
#ifdef NTFS_FOR_NT4
#include "ntfsboot4.inc"
#else
#include "ntfsboot.inc"
#endif
	};

	static const BYTE sc_NtfsRootDir[] = {
#include "ntfsroot.inc"
	};

	static const BYTE sc_NtfsMftRle[] = {
#include "ntfsmft.inc"
	};

	// Allocate 64KB from heap for MFT decompression, etc
	// NT4 allows this to be 16KB, NT 3.5x does not.
#ifdef NTFS_FOR_NT4
#define MFT_OFFSET_FROM_1KB_TO_4KB(Offset) Offset
#else
#define MFT_OFFSET_FROM_1KB_TO_4KB(Offset) (((Offset) % 0x400) + (((Offset) / 0x400) * 0x1000))
#endif
	PBYTE pMft = (PBYTE)malloc(0x4000);
	if (pMft == NULL) return _ENOMEM;

	// Decompress RLE compressed MFT to allocated buffer
	RppDecodeRle(sc_NtfsMftRle, sizeof(sc_NtfsMftRle), pMft, 0x4000);

#ifndef NTFS_FOR_NT4
	// Convert MFT from 1024 byte entries to 4KB entries.
	enum {
		FILE_UPDATE_SEQUENCE_OFF_OFFSET = 0x04,
		FILE_UPDATE_SEQUENCE_COUNT_OFFSET = 0x06,
		FILE_ATTRIBUTE_OFF_OFFSET = 0x14,
		FILE_HEADER_REAL_SIZE_OFFSET = 0x18,
		FILE_HEADER_ALLOCATED_SIZE_OFFSET = 0x1C,
	};
	PBYTE pMft4K = (PBYTE)malloc(MFT_OFFSET_FROM_1KB_TO_4KB(0x4000));
	if (pMft4K == NULL) {
		free(pMft);
		return _ENOMEM;
	}
	memset(pMft4K, 0, MFT_OFFSET_FROM_1KB_TO_4KB(0x4000));
	for (
		int inOff = 0, outOff = 0;
		inOff < 0x4000;
		inOff += 0x400, outOff += MFT_OFFSET_FROM_1KB_TO_4KB(0x400)
	) {
		memcpy(&pMft4K[outOff], &pMft[inOff], 0x400);
		U32LE mftSize = { .v = MFT_OFFSET_FROM_1KB_TO_4KB(0x400) };
		memcpy(&pMft4K[outOff + FILE_HEADER_ALLOCATED_SIZE_OFFSET], (PBYTE)(ULONG)&mftSize, sizeof(mftSize));
		// Expand the update sequence by 6 entries to 9 to take into account the extra allocated size.
		// Including 64-bit alignment, this is another 0x10 bytes all zerofilled.
		U32LE temp32 = { .v = 0 };
		U16LE temp16 = { .v = 0 };
		// 1) get offset to attribute data.
		memcpy((PBYTE)(ULONG)&temp16, &pMft4K[outOff + FILE_ATTRIBUTE_OFF_OFFSET], sizeof(temp16));
		// 2) copy all data up by 0x10 bytes, zerofill the bytes left behind
		memmove(&pMft4K[outOff + temp16.v + 0x10], &pMft4K[outOff + temp16.v], 0x400 - temp16.v);
		memset(&pMft4K[outOff + temp16.v], 0, 0x10);
		// 3) fix up offsets, lengths and counts
		temp16.v += 0x10;
		memcpy(&pMft4K[outOff + FILE_ATTRIBUTE_OFF_OFFSET], (PBYTE)(ULONG)&temp16, sizeof(temp16));

		temp16.v = 9;
		memcpy(&pMft4K[outOff + FILE_UPDATE_SEQUENCE_COUNT_OFFSET], (PBYTE)(ULONG)&temp16, sizeof(temp16));

		memcpy((PBYTE)(ULONG)&temp32, &pMft4K[outOff + FILE_HEADER_REAL_SIZE_OFFSET], sizeof(temp32));
		temp32.v += 0x10;
		memcpy(&pMft4K[outOff + FILE_HEADER_REAL_SIZE_OFFSET], (PBYTE)(ULONG)&temp32, sizeof(temp32));
		// 4) make sure u16 usn[0] is at end of each sector
		memcpy((PBYTE)(ULONG)&temp16, &pMft4K[outOff + FILE_UPDATE_SEQUENCE_OFF_OFFSET], sizeof(temp16));
		if ((ULONG)temp16.v >= temp32.v) {
			// invalid MFT?!
			free(pMft4K);
			free(pMft);
			return _EBADF;
		}
		memcpy((PBYTE)(ULONG)&temp16, &pMft4K[outOff + temp16.v], sizeof(temp16));

		for (ULONG offUsn = 0; offUsn < MFT_OFFSET_FROM_1KB_TO_4KB(0x400); offUsn += 0x200) {
			memcpy(&pMft4K[outOff + offUsn + 0x1FE], (PBYTE)(ULONG)&temp16, sizeof(temp16));
		}
	}
	free(pMft);
	pMft = pMft4K;
#endif

	// Allocate space for empty cluster
	BYTE EmptyCluster[0x1000] = { 0 };

	// Allocate space from stack for boot sector and copy from ntfsboot
	BYTE BootSector[0x200];
	memcpy(BootSector, sc_NtfsBoot, sizeof(BootSector));

	// Allocate space from stack for root directory
	BYTE RootDir[0x1000];
	memcpy(RootDir, sc_NtfsRootDir, sizeof(sc_NtfsRootDir));

	enum {
		NTFSBOOT_OFFSET_SIZE = 0x28,
		NTFSBOOT_OFFSET_BACKUP_MFT = 0x38,
		NTFSBOOT_OFFSET_VOLUME_SERIAL = 0x48
	};

	// ntfsboot.inc hardcodes a cluster size of 4KB, that is, 
	int64_t PartitionSizeInSectors = ( ((int64_t)SizeMb) * REPART_MB_SECTORS) - 1;

	{
		LARGE_INTEGER PartitionSizeInSectorsLi = { .QuadPart = PartitionSizeInSectors };
		memcpy(&BootSector[NTFSBOOT_OFFSET_SIZE], (void*)(ULONG)&PartitionSizeInSectorsLi, sizeof(PartitionSizeInSectors));
	}

	// Calculate the offset to the backup MFT in clusters. That is: sector count / (2 * 8), where 8 is number of sectors per cluster.
	int64_t BackupMftCluster = PartitionSizeInSectors / (2 * 8);
	// The partition size in clusters is exactly two times the length of this.
	int64_t PartitionSizeInClusters = BackupMftCluster * 2;

	{
		LARGE_INTEGER BackupMftClusterLi = { .QuadPart = BackupMftCluster };
		memcpy(&BootSector[NTFSBOOT_OFFSET_BACKUP_MFT], (void*)(ULONG)&BackupMftClusterLi, sizeof(BackupMftClusterLi));
	}

	// Calculate the new volume serial.
	// Use the same hashing algorithm to calculate the upper half as NT itself does; but for the low part use the bitwise NOT of the MBR signature.
	{
		LARGE_INTEGER NewVolumeSerial;
		NewVolumeSerial.LowPart = ~MbrSig;

		U32LE volid = { .v = MbrSig };

		PUCHAR pNvs = (PUCHAR)(ULONG)&volid;
		for (ULONG i = 0; i < sizeof(ULONG); i++) {
			volid.v += *pNvs++;
			volid.v = (volid.v >> 2) + (volid .v<< 30);
		}
		NewVolumeSerial.HighPart = volid.v;
		memcpy(&BootSector[NTFSBOOT_OFFSET_VOLUME_SERIAL], (void*)(ULONG)&NewVolumeSerial, sizeof(NewVolumeSerial));
	}

	// Calculate the offsets for each file.
	enum {

#ifdef NTFS_FOR_NT4
		MFT_MFTMIRR_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x5A8),
		MFT_LOGFILE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x9A8),
		MFT_ATTRDEF_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x11A8),
		MFT_ROOTDIR_INDEX_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x15E0),
		MFT_BITMAP_DISKLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x18B8),
		MFT_BITMAP_REALLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x18C0),
		MFT_BITMAP_CLUSLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1978),
		MFT_BITMAP_REALSIZE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1988),
		MFT_BITMAP_FILESIZE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1990),
		MFT_BITMAP_VALIDLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1998),
		MFT_BITMAP_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x19A0),
		MFT_BADCLUS_CLUS64_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x2198),
		MFT_BADCLUS_DISKLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x21A8),
		MFT_BADCLUS_REALLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x21B0),
		MFT_BADCLUS_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x21C8),
		MFT_UPCASE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x29A0),

		ROOTDIR_DISKLEN_OFFSET = 0x160,
		ROOTDIR_REALLEN_OFFSET = 0x168,

		MFT_BACKUP_CLUSTERS = 1,
#else
		MFT_MFT_DISKLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0xB8 + 0x10), // 0x10000
		MFT_MFT_REALLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0xC0 + 0x10), // 0x10000
		MFT_MFT_CLUSLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x178 + 0x10), // 0x0F
		MFT_MFT_REALSIZE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x188 + 0x10), // 0x10000
		MFT_MFT_FILESIZE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x190 + 0x10), // 0x10000
		MFT_MFT_VALIDLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x198 + 0x10), // 0x10000
		MFT_MFT_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1A0 + 0x10),
		MFT_MFTMIRR_DISKLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x4B8 + 0x10), // 0x4000
		MFT_MFTMIRR_REALLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x4C0 + 0x10), // 0x4000
		MFT_MFTMIRR_CLUSLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x580 + 0x10), // 3
		MFT_MFTMIRR_REALSIZE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x590 + 0x10), // 0x4000
		MFT_MFTMIRR_FILESIZE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x598 + 0x10), // 0x4000
		MFT_MFTMIRR_VALIDLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x5A0 + 0x10), // 0x4000
		MFT_MFTMIRR_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x5A8 + 0x10),
		MFT_LOGFILE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x9A8 + 0x10),
		MFT_ATTRDEF_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x11A8 + 0x10),
		MFT_ROOTDIR_INDEX_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x15E0 + 0x10),
		MFT_BITMAP_DISKLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x18B8 + 0x10),
		MFT_BITMAP_REALLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x18C0 + 0x10),
		MFT_BITMAP_CLUSLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1978 + 0x10),
		MFT_BITMAP_REALSIZE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1988 + 0x10),
		MFT_BITMAP_FILESIZE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1990 + 0x10),
		MFT_BITMAP_VALIDLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x1998 + 0x10),
		MFT_BITMAP_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x19A0 + 0x10),
		MFT_BADCLUS_CLUS64_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x2198 + 0x10),
		MFT_BADCLUS_DISKLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x21A8 + 0x10),
		MFT_BADCLUS_REALLEN_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x21B0 + 0x10),
		MFT_BADCLUS_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x21C8 + 0x10),
		MFT_UPCASE_OFFSET = MFT_OFFSET_FROM_1KB_TO_4KB(0x29A0 + 0x10),

		ROOTDIR_DISKLEN_OFFSET = 0x160,
		ROOTDIR_REALLEN_OFFSET = 0x168,

		ROOTDIR_MFT_DISKLEN_OFFSET = 0x288,
		ROOTDIR_MFT_REALLEN_OFFSET = 0x290,

		ROOTDIR_MFTMIRR_DISKLEN_OFFSET = 0x288,
		ROOTDIR_MFTMIRR_REALLEN_OFFSET = 0x290,

		MFT_BACKUP_CLUSTERS = 4,
#endif
	};

	ARC_STATUS Status = _ESUCCESS;

	// Calculate the number of clusters for $Bitmap.
	// PartitionSizeInClusters / 8.
	int64_t BitmapCountBytes = (PartitionSizeInClusters / 8);
	if ((PartitionSizeInClusters & 7) != 0) BitmapCountBytes++;
	int32_t BitmapCountClusters = (ULONG)(BitmapCountBytes / 0x1000);
	if ((BitmapCountBytes & 0xFFF) != 0) BitmapCountClusters++;
	LARGE_INTEGER BitmapRealSize = { .QuadPart = BitmapCountClusters };
	BitmapRealSize.QuadPart *= 0x1000;
	LARGE_INTEGER BitmapDiskSize = { .QuadPart = BitmapCountBytes };
#ifndef NTFS_FOR_NT4
	LARGE_INTEGER MftSize = { .QuadPart = MFT_OFFSET_FROM_1KB_TO_4KB(0x4000) };
	LARGE_INTEGER MftMirrSize = { .QuadPart = MFT_OFFSET_FROM_1KB_TO_4KB(0x1000) };
#endif
	do {
		// MftMirr is at BackupMftCluster.
		if (!RppMftWriteCluster24(pMft, MFT_MFTMIRR_OFFSET, BackupMftCluster)) {
			//printf("bad backup: %08x\r\n", (ULONG)BackupMftCluster);
			Status = _E2BIG;
			break;
		}

		int64_t CurrentCluster = BackupMftCluster + MFT_BACKUP_CLUSTERS;

		// LogFile is after BackupMft.
		if (!RppMftWriteCluster24(pMft, MFT_LOGFILE_OFFSET, CurrentCluster)) {
			//printf("bad logfile: %08x\r\n", (ULONG)CurrentCluster);
			Status = _E2BIG;
			break;
		}

		// AttrDef is after LogFile
		CurrentCluster += 1024;
		if (!RppMftWriteCluster24(pMft, MFT_ATTRDEF_OFFSET, CurrentCluster)) {
			//printf("bad attrdef: %08x\r\n", (ULONG)CurrentCluster);
			Status = _E2BIG;
			break;
		}

		// Root directory index is after AttrDef
		CurrentCluster += 9;
		if (!RppMftWriteCluster24(pMft, MFT_ROOTDIR_INDEX_OFFSET, CurrentCluster)) {
			//printf("bad rootdir: %08x\r\n", (ULONG)CurrentCluster);
			Status = _E2BIG;
			break;
		}

		// Bitmap is after root directory index
		CurrentCluster += 1;
		if (!RppMftWriteCluster24(pMft, MFT_BITMAP_OFFSET, CurrentCluster)) {
			//printf("bad bitmap: %08x\r\n", (ULONG)CurrentCluster);
			Status = _E2BIG;
			break;
		}

		// Also copy in the new filesizes for the bitmap file, into the MFT and root directory
		// note: BitmapDiskSize is size, BitmapRealSize is "size on disk" (ie, multiple of clusters)
		memcpy(&pMft[MFT_BITMAP_DISKLEN_OFFSET], (PVOID)(ULONG)&BitmapDiskSize, sizeof(BitmapDiskSize));
		memcpy(&pMft[MFT_BITMAP_REALLEN_OFFSET], (PVOID)(ULONG)&BitmapRealSize, sizeof(BitmapRealSize));
		memcpy(&pMft[MFT_BITMAP_REALSIZE_OFFSET], (PVOID)(ULONG)&BitmapRealSize, sizeof(BitmapRealSize));
		memcpy(&pMft[MFT_BITMAP_FILESIZE_OFFSET], (PVOID)(ULONG)&BitmapDiskSize, sizeof(BitmapDiskSize));
		memcpy(&pMft[MFT_BITMAP_VALIDLEN_OFFSET], (PVOID)(ULONG)&BitmapDiskSize, sizeof(BitmapDiskSize));
		memcpy(&RootDir[ROOTDIR_DISKLEN_OFFSET], (PVOID)(ULONG)&BitmapDiskSize, sizeof(BitmapDiskSize));
		memcpy(&RootDir[ROOTDIR_REALLEN_OFFSET], (PVOID)(ULONG)&BitmapRealSize, sizeof(BitmapRealSize));

		// Need to also specify the new length in MFT for bitmap.
		if (BitmapCountClusters >= 0x100) {
			// This is for a single byte, which implies ~32GB partition limit.
			// Just don't bother supporting this for now as we have 8GB limit on NT partition anyway:
			// To support this, we would have to increase the header byte and move the other bytes away enough space.
			Status = _E2BIG;
			break;
		}
		pMft[MFT_BITMAP_OFFSET + 1] = (BYTE)BitmapCountClusters;
		// BUGBUG: this is u64, but cluster count fits in a byte, that was just checked.
		pMft[MFT_BITMAP_CLUSLEN_OFFSET] = (BYTE)BitmapCountClusters - 1;

#ifndef NTFS_FOR_NT4
		// Need to fix up the lengths of primary and backup MFTs.
		memcpy(&pMft[MFT_MFT_DISKLEN_OFFSET], (PVOID)(ULONG)&MftSize, sizeof(MftSize));
		memcpy(&pMft[MFT_MFT_REALLEN_OFFSET], (PVOID)(ULONG)&MftSize, sizeof(MftSize));
		memcpy(&pMft[MFT_MFT_REALSIZE_OFFSET], (PVOID)(ULONG)&MftSize, sizeof(MftSize));
		memcpy(&pMft[MFT_MFT_FILESIZE_OFFSET], (PVOID)(ULONG)&MftSize, sizeof(MftSize));
		memcpy(&pMft[MFT_MFT_VALIDLEN_OFFSET], (PVOID)(ULONG)&MftSize, sizeof(MftSize));
		memcpy(&RootDir[ROOTDIR_DISKLEN_OFFSET], (PVOID)(ULONG)&MftSize, sizeof(MftSize));
		memcpy(&RootDir[ROOTDIR_REALLEN_OFFSET], (PVOID)(ULONG)&MftSize, sizeof(MftSize));

		pMft[MFT_MFT_OFFSET + 1] = 0x10;
		pMft[MFT_MFT_CLUSLEN_OFFSET] = 0x10 - 1;

		memcpy(&pMft[MFT_MFTMIRR_DISKLEN_OFFSET], (PVOID)(ULONG)&MftMirrSize, sizeof(MftMirrSize));
		memcpy(&pMft[MFT_MFTMIRR_REALLEN_OFFSET], (PVOID)(ULONG)&MftMirrSize, sizeof(MftMirrSize));
		memcpy(&pMft[MFT_MFTMIRR_REALSIZE_OFFSET], (PVOID)(ULONG)&MftMirrSize, sizeof(MftMirrSize));
		memcpy(&pMft[MFT_MFTMIRR_FILESIZE_OFFSET], (PVOID)(ULONG)&MftMirrSize, sizeof(MftMirrSize));
		memcpy(&pMft[MFT_MFTMIRR_VALIDLEN_OFFSET], (PVOID)(ULONG)&MftMirrSize, sizeof(MftMirrSize));
		memcpy(&RootDir[ROOTDIR_DISKLEN_OFFSET], (PVOID)(ULONG)&MftMirrSize, sizeof(MftMirrSize));
		memcpy(&RootDir[ROOTDIR_REALLEN_OFFSET], (PVOID)(ULONG)&MftMirrSize, sizeof(MftMirrSize));

		pMft[MFT_MFTMIRR_OFFSET + 1] = 4;
		pMft[MFT_MFTMIRR_CLUSLEN_OFFSET] = 4 - 1;
#endif

		// Bad clusters which specifies the whole disk.
		if (!RppMftWriteBadCluster24(pMft, MFT_BADCLUS_OFFSET, PartitionSizeInClusters + 1)) {
			//printf("bad badclus: %08x\r\n", (ULONG)CurrentCluster);
			Status = _E2BIG;
			break;
		}

		// Also copy in the partition size in clusters to the other place in that MFT entry.
		{
			LARGE_INTEGER PartSizeClus = { .QuadPart = PartitionSizeInClusters };
			memcpy(&pMft[MFT_BADCLUS_CLUS64_OFFSET], (PVOID)(ULONG)&PartSizeClus, sizeof(PartSizeClus));
			// Size on disk + size of file needs to be equal to the partition size in bytes
			// This implies a partition size limit of 1EB, good luck with putting that in an MBR though
			PartSizeClus.QuadPart *= 0x1000;
			memcpy(&pMft[MFT_BADCLUS_DISKLEN_OFFSET], (PVOID)(ULONG)&PartSizeClus, sizeof(PartSizeClus));
			memcpy(&pMft[MFT_BADCLUS_REALLEN_OFFSET], (PVOID)(ULONG)&PartSizeClus, sizeof(PartSizeClus));
		}

		// Uppercase table is after bitmap.
		CurrentCluster += BitmapCountClusters;
		if (!RppMftWriteCluster24(pMft, MFT_UPCASE_OFFSET, CurrentCluster)) {
			//printf("bad upcase: %08x\r\n", (ULONG)CurrentCluster);
			Status = _E2BIG;
			break;
		}
	} while (false);

	if (ARC_FAIL(Status)) {
		free(pMft);
		return Status;
	}

	// MFT has been created for this partition.

	// Now we start writing.
	int64_t SectorOffset64 = StartSector;
	SectorOffset64 *= REPART_SECTOR_SIZE;

	do {
		// Seek to the partition start.
		LARGE_INTEGER SectorOffset = { .QuadPart = SectorOffset64 };
		Status = Vectors->Seek(DeviceId, &SectorOffset, SeekAbsolute);
		if (ARC_FAIL(Status)) break;

		// Write the boot sector, followed by the remainder of the boot code
		ULONG Count = 0;
		Status = Vectors->Write(DeviceId, BootSector, sizeof(BootSector), &Count);
		if (ARC_SUCCESS(Status) && Count != sizeof(BootSector)) Status = _EIO;
		if (ARC_FAIL(Status)) break;
		Status = Vectors->Write(DeviceId, &sc_NtfsBoot[sizeof(BootSector)], sizeof(sc_NtfsBoot) - sizeof(BootSector), &Count);
		if (ARC_SUCCESS(Status) && Count != (sizeof(sc_NtfsBoot) - sizeof(BootSector))) Status = _EIO;
		if (ARC_FAIL(Status)) break;
		Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
		if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
		if (ARC_FAIL(Status)) break;

		// Write the used bitmap for the primary MFT. This is 64k clusters (256MB), with the first 16 clusters being used
		EmptyCluster[0] = EmptyCluster[1] = 0xFF;
		Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
		if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
		if (ARC_FAIL(Status)) break;
		EmptyCluster[0] = EmptyCluster[1] = 0x00;
		Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
		if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
		if (ARC_FAIL(Status)) break;
		
		// Write the primary MFT.
		Status = Vectors->Write(DeviceId, pMft, MFT_OFFSET_FROM_1KB_TO_4KB(0x4000), &Count);
		if (ARC_SUCCESS(Status) && Count != MFT_OFFSET_FROM_1KB_TO_4KB(0x4000)) Status = _EIO;
		if (ARC_FAIL(Status)) break;

		// Seek to the backup MFT location.
		SectorOffset.QuadPart += (BackupMftCluster * 0x1000);
		Status = Vectors->Seek(DeviceId, &SectorOffset, SeekAbsolute);
		if (ARC_FAIL(Status)) break;

		// Write the backup MFT, which is only the first 4 elements of the primary MFT.
		Status = Vectors->Write(DeviceId, pMft, MFT_OFFSET_FROM_1KB_TO_4KB(0x1000), &Count);
		if (ARC_SUCCESS(Status) && Count != MFT_OFFSET_FROM_1KB_TO_4KB(0x1000)) Status = _EIO;
		if (ARC_FAIL(Status)) break;

		// Next up is logfile which is 4MB FF-filled
		memset(EmptyCluster, 0xFF, sizeof(EmptyCluster));
		for (ULONG i = 0; i < 0x400000 / 0x1000; i++) {
			Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
			if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
			if (ARC_FAIL(Status)) break;
		}
		memset(EmptyCluster, 0, sizeof(EmptyCluster));

		// Next up is AttrDef
		Status = RppWriteAttrDef(DeviceId, Vectors);
		if (ARC_FAIL(Status)) break;

		// Next up is root directory index
		Status = Vectors->Write(DeviceId, RootDir, sizeof(RootDir), &Count);
		if (ARC_SUCCESS(Status) && Count != sizeof(RootDir)) Status = _EIO;
		if (ARC_FAIL(Status)) break;

		// Next up is bitmap.
		// This bitmap starts at the MFT cluster (cluster 4)
		// One bit set per used cluster.

		{
			// First, write the initial bitmap cluster. This represents 32768 clusters.
			// Cluster 0 "unused" (really bootdata, nothing will write to that), then next 7 (for NT4) or 28 (for NT3.x) clusters are used by the MFT so set that.
#ifdef NTFS_FOR_NT4
			EmptyCluster[0] = 0xF7;
#else
			EmptyCluster[0] = EmptyCluster[1] = 0xFF;
			EmptyCluster[2] = 0x07;
#endif
			Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
			if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
			if (ARC_FAIL(Status)) break;
			EmptyCluster[0] = 0;

			// Now we need to loop until we hit BackupMftCluster.
			ULONG BitmapCurrentCluster = (sizeof(EmptyCluster) * 8);
			ULONG Offset = ((ULONG)BackupMftCluster) - BitmapCurrentCluster;
			while (Offset > (sizeof(EmptyCluster) * 8)) {

				Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
				if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
				if (ARC_FAIL(Status)) break;

				BitmapCurrentCluster += (sizeof(EmptyCluster) * 8);
				Offset = ((ULONG)BackupMftCluster) - BitmapCurrentCluster;
			}

			if (ARC_FAIL(Status)) break;

			// At some point in the following cluster, is the BackupMftCluster.
			// Calculate the total used clusters.
			ULONG UsedClusters = 1 + 1024 + 9 + 1 + BitmapCountClusters + 32;

			// Find the bit offset inside EmptyCluster to start setting bits.
			ULONG ByteOffset = Offset / 8;
			ULONG BitOffset = Offset & 7;

			// Start setting bits.
			while (UsedClusters != 0) {
				while (UsedClusters != 0 && ByteOffset < sizeof(EmptyCluster)) {
					EmptyCluster[ByteOffset] |= (1 << BitOffset);

					BitOffset++;
					if (BitOffset == 8) {
						BitOffset = 0;
						ByteOffset++;
					}
					UsedClusters--;
				}
				// Set as many bits as possible in this loop.
				// Write this cluster to disk.
				Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
				if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
				if (ARC_FAIL(Status)) break;
				// And set back to zero.
				memset(EmptyCluster, 0, sizeof(EmptyCluster));
				BitmapCurrentCluster += (sizeof(EmptyCluster) * 8);
				ByteOffset = 0;
				BitOffset = 0;
			}

			if (ARC_FAIL(Status)) break;

			// Written out the used clusters that have been used halfway through the disk.
			// Now loop until we hit PartitionSizeInClusters.
			Offset = ((ULONG)PartitionSizeInClusters) - BitmapCurrentCluster;
			while (Offset > (sizeof(EmptyCluster) * 8)) {

				Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
				if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
				if (ARC_FAIL(Status)) break;

				BitmapCurrentCluster += (sizeof(EmptyCluster) * 8);
				Offset = ((ULONG)PartitionSizeInClusters) - BitmapCurrentCluster;
			}

			// Reached the final cluster.
			// Find the bit offset inside EmptyCluster to start setting bits.
			ByteOffset = Offset / 8;
			BitOffset = Offset & 7;

			// Every bit starting from this offset needs to be set.
			if (BitOffset != 0) {
				while (BitOffset < 8) {
					EmptyCluster[ByteOffset] |= (1 << BitOffset);
					BitOffset++;
				}
				ByteOffset++;
			}
			// And set the remainder of the bits in the bitmap.
			memset(&EmptyCluster[ByteOffset], 0xFF, sizeof(EmptyCluster) - ByteOffset);
			// And write out the final cluster.
			Status = Vectors->Write(DeviceId, EmptyCluster, sizeof(EmptyCluster), &Count);
			if (ARC_SUCCESS(Status) && Count != sizeof(EmptyCluster)) Status = _EIO;
			if (ARC_FAIL(Status)) break;
			// And set back to zero.
			memset(EmptyCluster, 0, sizeof(EmptyCluster));
		}

		// Write the uppercase table.
		Status = RppWriteUpCaseTable(DeviceId, Vectors);
		if (ARC_FAIL(Status)) break;

		// Seek to the final sector of this partition.
		SectorOffset.QuadPart = SectorOffset64;
		SectorOffset.QuadPart += ((int64_t)SizeMb * 0x100000) - REPART_SECTOR_SIZE;
		Status = Vectors->Seek(DeviceId, &SectorOffset, SeekAbsolute);
		if (ARC_FAIL(Status)) break;

		// Write the backup boot sector there.
		Status = Vectors->Write(DeviceId, BootSector, sizeof(BootSector), &Count);
		if (ARC_SUCCESS(Status) && Count != sizeof(BootSector)) Status = _EIO;
		if (ARC_FAIL(Status)) break;

		// All done!
	} while (false);

	free(pMft);
	return Status;
}

/// <summary>
/// Repartitions a USB disk, writing an MBR partition table, ARC environment space, an NT partition and an ARC system partition.
/// </summary>
/// <param name="DeviceId">Device ID</param>
/// <param name="NtPartMb">Disk space in MB for the NT partition</param>
/// <param name="DataWritten">If failed, will be set to true after data has been written to disk.</param>
/// <returns>ARC status code.</returns>
ARC_STATUS ArcFsRepartitionDisk(ULONG DeviceId, ULONG NtPartMb, bool* DataWritten) {
	// This does repartition the disk for NT.
	// Creates the following MBR partition table:
	// Partition 1 - type 0x41, start sector 1, size 2 sectors, this is for ARC environment
	// Partition 2 - FAT16 or NTFS, start 1MB, size "NtPartMb"MB, this is the NT system partition. If below 2GB in size, leave unformatted, otherwise, format as NTFS.
	// Partition 3 - FAT16, ARC system partition, start after OS partition, size 32MB.

	if (DataWritten == NULL) return _EINVAL;
	*DataWritten = false;
	// If the NT partition is over the maximum allowed size, do nothing
	if (NtPartMb > REPART_MAX_NT_PART_IN_MB) return _E2BIG;

	// Get the device.
	PARC_FILE_TABLE Device = ArcIoGetFile(DeviceId);
	if (Device == NULL) return _EBADF;
	// Can't be a file.
	if (Device->DeviceId != FILE_IS_RAW_DEVICE) return _EBADF;
	// Sector size must be 512 bytes. atapi.sys expects this anyway!
	if (Device->GetSectorSize == NULL) return _EBADF;
	ULONG SectorSize;
	if (ARC_FAIL(Device->GetSectorSize(DeviceId, &SectorSize))) return _EBADF;
	if (SectorSize != REPART_SECTOR_SIZE) return _EBADF;
	PDEVICE_VECTORS Vectors = Device->DeviceEntryTable;
	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();

	// Size of disk must be able to fit the partition table that we're attempting to write out
	ULONG TotalSizeMb = 1 // MBR partition 1
		+ 32 // MBR partition 2
		+ NtPartMb // MBR partition 3
		;
	
	// Calculate the disk size in MB, if the total partitions size is greater than that, error.
	FILE_INFORMATION FileInfo;
	if (ARC_FAIL(Vectors->GetFileInformation(DeviceId, &FileInfo))) return _EBADF;
	ULONG DiskSizeMb = (ULONG)(FileInfo.EndingAddress.QuadPart / 0x100000);
	if (TotalSizeMb > DiskSizeMb) return _E2BIG;

	// For the MBR disk signature, use CRC32 of TIME_FIELDS structure. Accurate to the second, but that's the best platform-independent entropy source we have right now...
	PTIME_FIELDS Time = Api->GetTimeRoutine();

	// Initialise the MBR sector.
	printf("Initialising partition tables...\r\n");
	MBR_SECTOR Mbr = { 0 };

	// MBR itself
	bool NtPartitionIsExtended = (NtPartMb > 2048); // if NT partition is over 2GB in size, format as NTFS
	Mbr.ValidMbr = MBR_VALID_SIGNATURE;
	Mbr.Partitions[0].Type = 0x41;
	Mbr.Partitions[0].SectorStart = REPART_MBR_PART1_START;
	Mbr.Partitions[0].SectorCount = REPART_MBR_PART1_SIZE;
	Mbr.Partitions[1].Type = NtPartitionIsExtended ? 0x07 : 0x06;
	Mbr.Partitions[1].SectorStart = REPART_MBR_PART2_START;
	Mbr.Partitions[1].SectorCount = NtPartMb * REPART_MB_SECTORS;
	Mbr.Partitions[2].Type = 0x06;
	Mbr.Partitions[2].SectorStart = Mbr.Partitions[1].SectorStart + Mbr.Partitions[1].SectorCount;
	Mbr.Partitions[2].SectorCount = REPART_MBR_PART3_SIZE;
	ULONG MbrSignature = Crc32(Time, sizeof(*Time));
	// Disallow an all-zero MBR signature.
	if (MbrSignature == 0) MbrSignature = 0xFFFFFFFFul;
	Mbr.Signature = MbrSignature;
	ULONG ArcSystemPartitionSectorOffset = (NtPartMb * REPART_MB_SECTORS) + REPART_MBR_PART2_START;

	// Allocate heap space for laying out the FAT FS for the ARC system partition
	enum {
		SIZE_OF_SYS_PART_FAT_FS = 0x14200
	};
	PUCHAR SysPartFatFs = (PUCHAR)malloc(SIZE_OF_SYS_PART_FAT_FS);
	if (SysPartFatFs == NULL) {
		printf("Could not allocate memory for FAT filesystem\r\n");
		return _ENOMEM;
	}
	memset(SysPartFatFs, 0, SIZE_OF_SYS_PART_FAT_FS);


	ARC_STATUS Status = _ESUCCESS;
	do {
		// Write everything to disk.
		// This is where overwriting existing data on the disk starts!
		LARGE_INTEGER SeekOffset = INT32_TO_LARGE_INTEGER(0);
		Status = Vectors->Seek(DeviceId, &SeekOffset, SeekAbsolute);
		if (ARC_FAIL(Status)) break;
		// Seek successful so perform the write. Print progress now (after seek), as we are about to start committing to writing this disk layout out:
		printf("Writing partition tables...\r\n");
		*DataWritten = true;
		// MBR first
		ULONG Count = 0;
		Status = Vectors->Write(DeviceId, &Mbr, sizeof(Mbr), &Count);
		if (ARC_FAIL(Status) || Count != sizeof(Mbr)) {
			printf("Could not write DDT and MBR partition table\r\n");
			if (ARC_SUCCESS(Status)) Status = _EIO;
			break;
		}

		// Current position: partition 1 (ARC environment space)
		printf("Formatting ARC non-volatile storage...\r\n");
		memset(&Mbr, 0, sizeof(Mbr));
		for (ULONG Sector = 0; ARC_SUCCESS(Status) && Sector < 2; Sector++) {
			Count = 0;
			Status = Vectors->Write(DeviceId, &Mbr, sizeof(Mbr), &Count);
			if (ARC_SUCCESS(Status) && Count != sizeof(Mbr)) Status = _EIO;
		}
		if (ARC_FAIL(Status)) {
			printf("Could not format ARC non-volatile storage\r\n");
			break;
		}

		// Seek to MBR partition 3 and write the empty FAT filesystem there
		printf("Formatting FAT16 ARC system partition...\r\n");
		static UCHAR s_Bpb32M[] = {
			  0xEB, 0xFE, 0x90, 0x4D, 0x53, 0x44, 0x4F, 0x53, 0x35, 0x2E,
			  0x30, 0x00, 0x02, 0x04, 0x01, 0x00, 0x02, 0x00, 0x02, 0x00,
			  0x00, 0xF8, 0x40, 0x00, 0x20, 0x00, 0x40, 0x00, 0x00, 0x00,
			  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x80, 0x00, 0x29, 0x15,
			  0x45, 0x14, 0x2B, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
			  0x20, 0x20, 0x20, 0x20, 0x46, 0x41, 0x54, 0x31, 0x36, 0x20,
			  0x20, 0x20
		};
		static UCHAR s_FatEmpty[] = { 0xF8, 0xFF, 0xFF, 0xFF };
		_Static_assert(__builtin_offsetof(MBR_SECTOR, MbrCode) == 0);
		memset(&Mbr, 0, sizeof(Mbr));
		memcpy(Mbr.MbrCode, s_Bpb32M, sizeof(s_Bpb32M));
		Mbr.ValidMbr = MBR_VALID_SIGNATURE;
		// Copy the boot sector
		memcpy(SysPartFatFs, &Mbr, sizeof(Mbr));
		// Copy the two copies of the FAT
		memset(&Mbr, 0, sizeof(Mbr));
		memcpy(Mbr.MbrCode, s_FatEmpty, sizeof(s_FatEmpty));
		memcpy(&SysPartFatFs[0x200], &Mbr, sizeof(Mbr));
		memcpy(&SysPartFatFs[0x8200], &Mbr, sizeof(Mbr));


		SeekOffset = INT32_TO_LARGE_INTEGER(ArcSystemPartitionSectorOffset);
		SeekOffset.QuadPart *= REPART_SECTOR_SIZE;
		Status = Vectors->Seek(DeviceId, &SeekOffset, SeekAbsolute);
		if (ARC_SUCCESS(Status)) {
			// Write to disk
			Count = 0;
			Status = Vectors->Write(DeviceId, SysPartFatFs, SIZE_OF_SYS_PART_FAT_FS, &Count);
			if (ARC_SUCCESS(Status) && Count != SIZE_OF_SYS_PART_FAT_FS) Status = _EIO;
		}
		if (ARC_FAIL(Status)) {
			printf("Could not format ARC system partition\r\n");
			break;
		}

		// Seek to MBR partition 2 and write zeroes over its first sector (this is enough to ensure no existing FAT/NTFS partition is there, right?)
		// If user specified size over 2GB, instead format as NTFS.
		memset(&Mbr, 0, sizeof(Mbr));
		ULONG PartitionBeingWiped = 0;
		if (NtPartitionIsExtended) {
			printf("Formatting NT OS partition as NTFS...\r\n");
			// Format as NTFS!
			Status = RpFormatNtfs(DeviceId, Vectors, REPART_MBR_PART2_START, NtPartMb, MbrSignature);
		}
		else {
			printf("Ensuring NT OS partition is considered unformatted...\r\n");
			SeekOffset = INT32_TO_LARGE_INTEGER(REPART_MBR_PART2_START * REPART_SECTOR_SIZE);
			Status = Vectors->Seek(DeviceId, &SeekOffset, SeekAbsolute);
			if (ARC_SUCCESS(Status)) {
				// Write to disk
				Count = 0;
				Status = Vectors->Write(DeviceId, &Mbr, sizeof(Mbr), &Count);
				if (ARC_SUCCESS(Status) && Count != sizeof(Mbr)) Status = _EIO;
			}
		}
		if (ARC_FAIL(Status)) {
			if (NtPartitionIsExtended) printf("Could not format NT OS partition\r\n");
			else printf("Could not clear initial sectors of NT partition\r\n");
			break;
		}

		// Everything is done
		printf("Successfully partitioned the disk.\r\n");
	} while (false);

	free(SysPartFatFs);
	return Status;
}

// Implement wrappers around libfat/libiso9660.

typedef enum {
	FS_UNKNOWN,
	FS_ISO9660,
	FS_FAT
} FS_TYPE;

enum {
	ISO9660_SECTOR_SIZE = 2048,
	FAT_SECTOR_SIZE = 512
};

typedef struct _FS_METADATA {
	union {
		struct {
			l9660_fs Iso9660;
			ULONG DeviceId;
		};
		struct {
			PFATFS Fat;
			ULONG SectorPresent;
			UCHAR Sector[FAT_SECTOR_SIZE];
			USHORT WritePosition;
			bool InWrite;
		};
	};
	FS_TYPE Type;
	ULONG SectorSize;
} FS_METADATA, *PFS_METADATA;
_Static_assert(FILE_TABLE_SIZE < 100);

static FS_METADATA s_Metadata[FILE_TABLE_SIZE] = { 0 };

//static ULONG s_CurrentDeviceId = FILE_IS_RAW_DEVICE;

DPSTATUS disk_initializep(void) { return RES_OK; }

DPRESULT disk_readp(ULONG DeviceId, BYTE* buff, DWORD sector, UINT offset, UINT count) {
	if (DeviceId >= FILE_TABLE_SIZE) return RES_PARERR;
	PFS_METADATA Meta = &s_Metadata[DeviceId];
	if (Meta->Type != FS_FAT) return RES_PARERR;
	// If a write transaction in progress, read will fail
	if (Meta->InWrite) return RES_ERROR;

	if (Meta->SectorPresent != sector) {
		PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
		// Seek to requested sector.
		int64_t Position64 = sector;
		Position64 *= FAT_SECTOR_SIZE;
		LARGE_INTEGER Position = Int64ToLargeInteger(Position64);

		ARC_STATUS Status = Api->SeekRoutine(DeviceId, &Position, SeekAbsolute);
		if (ARC_FAIL(Status)) {
			//printf("FAT: could not seek to sector %08x\n", sector);
			return RES_ERROR;
		}

		// Read to buffer.
		ULONG Length = sizeof(Meta->Sector);

		U32LE Count;
		Status = Api->ReadRoutine(DeviceId, Meta->Sector, Length, &Count);
		if (ARC_FAIL(Status)) {
			//printf("FAT: could not read sector %08x\n", sector);
			return RES_ERROR;
		}

		if (Count.v != sizeof(Meta->Sector)) return RES_ERROR;
		Meta->SectorPresent = sector;
	}

	memcpy(buff, &Meta->Sector[offset], count);
	return RES_OK;
}

DPRESULT disk_writep(ULONG DeviceId, const BYTE* buff, DWORD sc) {
	if (DeviceId >= FILE_TABLE_SIZE) return RES_PARERR;
	PFS_METADATA Meta = &s_Metadata[DeviceId];
	if (Meta->Type != FS_FAT) return RES_PARERR;

	if (buff == NULL) {
		if (!Meta->InWrite) {
			// Start a write to sector sc
			memset(Meta->Sector, 0, sizeof(Meta->Sector));
			Meta->SectorPresent = sc;
			Meta->InWrite = true;
			Meta->WritePosition = 0;
			return RES_OK;
		}

		if (sc == 0) return RES_PARERR; // sc must be 0 to finish write
		// Finish write, actually perform the write here:

		// Seek to requested sector.
		PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
		int64_t Position64 = Meta->SectorPresent;
		Position64 *= FAT_SECTOR_SIZE;

		LARGE_INTEGER Position = Int64ToLargeInteger(Position64);

		// Ensure any future reads will succeed
		Meta->SectorPresent = 0xffffffff;
		Meta->InWrite = false;

		ARC_STATUS Status = Api->SeekRoutine(DeviceId, &Position, SeekAbsolute);
		if (ARC_FAIL(Status)) {
			return RES_ERROR;
		}

		// Write to sector.
		ULONG Length = sizeof(Meta->Sector);

		U32LE Count;
		Status = Api->WriteRoutine(DeviceId, Meta->Sector, Length, &Count);
		if (ARC_FAIL(Status)) {
			return RES_ERROR;
		}

		if (Count.v != sizeof(Meta->Sector)) return RES_ERROR;
		return RES_OK;
	}

	// Write data into buffer
	DWORD afterOff = sc + Meta->WritePosition;
	if (afterOff > FAT_SECTOR_SIZE) return RES_PARERR;

	memcpy(&Meta->Sector[Meta->WritePosition], buff, sc);
	Meta->WritePosition += sc;
	return RES_OK;
}

static ARC_STATUS IsoErrorToArc(l9660_status status) {
	switch (status) {
	case L9660_OK:
		return _ESUCCESS;
	case L9660_EBADFS:
		return _EBADF;
	case L9660_EIO:
		return _EIO;
	case L9660_ENOENT:
		return _ENOENT;
	case L9660_ENOTDIR:
		return _ENOTDIR;
	case L9660_ENOTFILE:
		return _EISDIR;
	default:
		return _EFAULT;
	}
}

static ARC_STATUS FatErrorToArc(PFRESULT fr) {
	switch (fr) {
	case PFR_OK:
		return _ESUCCESS;
	case PFR_DISK_ERR:
		return _EIO;
	case PFR_NOT_READY:
		return _ENXIO;
	case PFR_NO_FILE:
		return _ENOENT;
	case PFR_NOT_OPENED:
		return _EINVAL;
	case PFR_NOT_ENABLED:
		return _ENXIO;
	case PFR_NO_FILESYSTEM:
		return _EBADF;
	default:
		return _EFAULT;
	}
}

static bool FsMediumIsoReadSectors(l9660_fs* fs, void* buffer, ULONG sector) {
	PFS_METADATA Metadata = (PFS_METADATA)fs;
	if (Metadata->Type != FS_ISO9660) return false;
	
	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
	// Seek to requested sector.
	int64_t Position64 = sector;
	Position64 *= ISO9660_SECTOR_SIZE;
	LARGE_INTEGER Position = Int64ToLargeInteger(Position64);

	ARC_STATUS Status = Api->SeekRoutine(Metadata->DeviceId, &Position, SeekAbsolute);
	if (ARC_FAIL(Status)) {
		printf("ISO: could not seek to sector %x\r\n", sector);
		return false;
	}

	// Read to buffer.
	ULONG Length = ISO9660_SECTOR_SIZE;

	U32LE Count;
	Status = Api->ReadRoutine(Metadata->DeviceId, buffer, Length, &Count);
	if (ARC_FAIL(Status)) {
		printf("ISO: could not read sector %x\r\n", sector);
		return false;
	}

	return Length == Count.v;
}

ARC_STATUS FsInitialiseForDevice(ULONG DeviceId) {
	if (DeviceId >= FILE_TABLE_SIZE) return _EBADF;
	PFS_METADATA FsMeta = &s_Metadata[DeviceId];

	// Obtain sector size for device.
	PARC_FILE_TABLE Device = ArcIoGetFile(DeviceId);
	if (Device == NULL) return _EBADF;
	if (Device->GetSectorSize == NULL) return _EBADF;
	ULONG SectorSize;
	ARC_STATUS Status = Device->GetSectorSize(DeviceId, &SectorSize);
	if (ARC_FAIL(Status)) return _EBADF;

	if (FsMeta->SectorSize == SectorSize) return _ESUCCESS;

	FsMeta->SectorSize = SectorSize;

	bool Mounted = false;
	if (SectorSize <= ISO9660_SECTOR_SIZE) {
		FsMeta->Type = FS_ISO9660;
		FsMeta->DeviceId = DeviceId;
		l9660_status IsoStatus = l9660_openfs(&FsMeta->Iso9660, FsMediumIsoReadSectors);
		Mounted = IsoStatus == L9660_OK;
	}
	if (!Mounted && SectorSize <= FAT_SECTOR_SIZE) {
		// ISO9660 mount failed, attempt FAT
		// Zero out everything used by the ISO9660 part
		memset(&FsMeta->Iso9660, 0, sizeof(FsMeta->Iso9660));
		FsMeta->DeviceId = 0;
		FsMeta->Type = FS_FAT;
		FsMeta->Fat.DeviceId = DeviceId;
		FsMeta->SectorPresent = 0xFFFFFFFF;
		Mounted = pf_mount(&FsMeta->Fat) == PFR_OK;
	}

	if (!Mounted) {
		memset(FsMeta, 0, sizeof(*FsMeta));
		return _EBADF;
	}

	return _ESUCCESS;
}

ARC_STATUS FsUnmountForDevice(ULONG DeviceId) {
	if (DeviceId >= FILE_TABLE_SIZE) return _EBADF;
	PFS_METADATA FsMeta = &s_Metadata[DeviceId];
	if (FsMeta->SectorSize == 0) return _EBADF;

	memset(FsMeta, 0, sizeof(*FsMeta));
	return _ESUCCESS;
}



// Filesystem device functions.
static ARC_STATUS FsOpen(PCHAR OpenPath, OPEN_MODE OpenMode, PULONG FileId) {
	// Only allow to open existing files.
	switch (OpenMode) {
	case ArcCreateDirectory:
	case ArcOpenDirectory:
	case ArcCreateReadWrite:
	case ArcCreateWriteOnly:
	case ArcSupersedeWriteOnly:
	case ArcSupersedeReadWrite:
		return _EINVAL;
	default: break;
	}

	//printf("Open %s(%d) [%x]\n", OpenPath, OpenMode, *FileId);
	// Get the file table, we need the device ID.
	PARC_FILE_TABLE File = ArcIoGetFileForOpen(*FileId);
	if (File == NULL) return _EBADF;
	ULONG DeviceId = File->DeviceId;
	PARC_FILE_TABLE Device = ArcIoGetFile(DeviceId);
	if (Device == NULL) return _EBADF;

	//s_CurrentDeviceId = DeviceId;
	PFS_METADATA Meta = &s_Metadata[DeviceId];

	ARC_STATUS Status = _ESUCCESS;

	//PCHAR EndOfPath = strrchr(OpenPath, '\\');
	File->u.FileContext.FileSize.QuadPart = 0;

	switch (Meta->Type) {
	case FS_ISO9660:
		// ISO9660 open file.
	{
		// Open root directory
		l9660_dir root;
		Status = IsoErrorToArc(l9660_fs_open_root(&root, &Meta->Iso9660));
		//printf("l9660_fs_open_root %d\r\n", Status);
		if (ARC_FAIL(Status)) return Status;

		// Open file.
		Status = IsoErrorToArc(l9660_openat(&File->u.FileContext.Iso9660, &root, &OpenPath[1]));
		//printf("l9660_fs_openat %d\r\n", Status);
		if (ARC_FAIL(Status)) return Status;

		File->u.FileContext.FileSize.LowPart = File->u.FileContext.Iso9660.length;

		break;
	}

	case FS_FAT:
		// FATFS open file.
		File->u.FileContext.Fat = Meta->Fat;
		Status = FatErrorToArc(pf_open(&File->u.FileContext.Fat, OpenPath));
		if (ARC_FAIL(Status)) return Status;

		File->u.FileContext.FileSize.LowPart = File->u.FileContext.Fat.fsize;
		break;

	default:
		return _EBADF;
	}

	// Set flags.
	switch (OpenMode) {
	case ArcOpenReadOnly:
		File->Flags.Read = 1;
		break;
	case ArcOpenWriteOnly:
		File->Flags.Write = 1;
		break;
	case ArcOpenReadWrite:
		File->Flags.Read = 1;
		File->Flags.Write = 1;
		break;
	default: break;
	}

	return Status;
}

static ARC_STATUS FsClose(ULONG FileId) {
	// Get the file table
	PARC_FILE_TABLE File = ArcIoGetFile(FileId);
	if (File == NULL) return _EBADF;

	PFS_METADATA Meta = &s_Metadata[File->DeviceId];
	
	switch (Meta->Type) {
	case FS_ISO9660:
		break;
	case FS_FAT:
		break;
	default:
		return _EBADF;
	}
	return _ESUCCESS;
}

static ARC_STATUS FsMount(PCHAR MountPath, MOUNT_OPERATION Operation) { return _EINVAL; }

static ARC_STATUS FsRead(ULONG FileId, PVOID Buffer, ULONG Length, PULONG Count) {
	//printf("Read %x => (%08x) %x bytes\r\n", FileId, Buffer, Length);
	// Get the file table
	PARC_FILE_TABLE File = ArcIoGetFile(FileId);
	if (File == NULL) return _EBADF;
	if (Length == 0) {
		*Count = 0;
		return _ESUCCESS;
	}

	PFS_METADATA Meta = &s_Metadata[File->DeviceId];

	switch (Meta->Type) {
	case FS_ISO9660:
		return IsoErrorToArc(l9660_read(&File->u.FileContext.Iso9660, Buffer, Length, Count));
	case FS_FAT:
		return FatErrorToArc(pf_read(&File->u.FileContext.Fat, Buffer, Length, Count));
	default:
		return _EBADF;
	}
}

static ARC_STATUS FsWrite(ULONG FileId, PVOID Buffer, ULONG Length, PULONG Count) {
	//printf("Write %x => %x bytes\n", FileId, Length);
	// Get the file table
	PARC_FILE_TABLE File = ArcIoGetFile(FileId);
	if (File == NULL) return _EBADF;

	PFS_METADATA Meta = &s_Metadata[File->DeviceId];
	switch (Meta->Type) {
	case FS_ISO9660:
		return _EBADF; // no writing to iso fs
	case FS_FAT:
		return FatErrorToArc(pf_write(&File->u.FileContext.Fat, Buffer, Length, Count));
	default:
		return _EBADF;
	}
}

static ARC_STATUS FsSeek(ULONG FileId, PLARGE_INTEGER Offset, SEEK_MODE SeekMode) {
	//printf("Seek %x => %llx (%d)\n", FileId, Offset->QuadPart, SeekMode);
	// Get the file table
	PARC_FILE_TABLE File = ArcIoGetFile(FileId);
	if (File == NULL) return _EBADF;
	// only support s32 offsets
	// positive
	if (Offset->QuadPart > INT32_MAX || Offset->QuadPart < INT32_MIN) {
		//printf("seek: Bad offset %llx\n", Offset->QuadPart);
		return _EINVAL;
	}

	int Origin;
	switch (SeekMode) {
	case SeekRelative:
		File->Position += Offset->QuadPart;
		if (File->Position > INT32_MAX) {
			//printf("seek: Bad offset %llx\n", Offset->QuadPart);
			return _EINVAL;
		}
		Origin = SEEK_CUR;
		break;
	case SeekAbsolute:
		File->Position = Offset->QuadPart;
		Origin = SEEK_SET;
		break;
	default:
		//printf("seek: Bad mode %d\n", SeekMode);
		return _EINVAL;
	}

	PFS_METADATA Meta = &s_Metadata[File->DeviceId];

	switch (Meta->Type) {
	case FS_ISO9660:
		return IsoErrorToArc(l9660_seek(&File->u.FileContext.Iso9660, Origin == SEEK_CUR ? L9660_SEEK_CUR : L9660_SEEK_SET, Offset->LowPart));
	case FS_FAT:
		return FatErrorToArc(pf_lseek(&File->u.FileContext.Fat, (ULONG)File->Position));
	default:
		return _EBADF;
	}
}

static ARC_STATUS FsGetFileInformation(ULONG FileId, PFILE_INFORMATION FileInfo) {
	// Get the file table
	PARC_FILE_TABLE File = ArcIoGetFile(FileId);
	if (File == NULL) return _EBADF;

	FileInfo->EndingAddress.QuadPart = File->u.FileContext.FileSize.QuadPart;
	FileInfo->CurrentPosition.QuadPart = File->Position;

	FileInfo->FileNameLength = 0; // TODO: fix?
	return _ESUCCESS;
}

static ARC_STATUS FsSetFileInformation(ULONG FileId, ULONG AttributeFlags, ULONG AttributeMask) {
	return _EACCES;
}

static ARC_STATUS FsGetDirectoryEntry(ULONG FileId, PDIRECTORY_ENTRY DirEntry, ULONG NumberDir, PULONG CountDir) {
	return _EBADF;
}

// Filesystem device vectors.
static const DEVICE_VECTORS FsVectors = {
	.Open = FsOpen,
	.Close = FsClose,
	.Mount = FsMount,
	.Read = FsRead,
	.Write = FsWrite,
	.Seek = FsSeek,
	.GetReadStatus = NULL,
	.GetFileInformation = FsGetFileInformation,
	.SetFileInformation = FsSetFileInformation,
	.GetDirectoryEntry = FsGetDirectoryEntry
};



void FsInitialiseTable(PARC_FILE_TABLE File) {
	ULONG DeviceId = File->DeviceId;
	PARC_FILE_TABLE Device = ArcIoGetFile(DeviceId);
	File->DeviceEntryTable = &FsVectors;
}