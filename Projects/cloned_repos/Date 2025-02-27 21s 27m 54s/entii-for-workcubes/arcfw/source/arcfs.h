#pragma once

enum {
	REPART_SECTOR_SIZE = 0x200,
	REPART_KB_SECTORS = 0x400 / REPART_SECTOR_SIZE,
	REPART_MB_SECTORS = 0x100000 / REPART_SECTOR_SIZE,
	REPART_U32_MAX_SECTORS_IN_MB = 0x1FFFFF,
	REPART_MAX_NT_PART_IN_MB = 8063 - 33, // CHS limit - 32MB for ARC system partition - 1MB for initial partitions + partition table


	REPART_MBR_PART1_START = 1,
	REPART_MBR_PART1_SIZE = (0x100000 / REPART_SECTOR_SIZE) - REPART_MBR_PART1_START,
	REPART_MBR_PART2_START = 0x100000 / REPART_SECTOR_SIZE,
	REPART_MBR_PART3_SIZE = REPART_MB_SECTORS * 32,
	REPART_MBR_CHS_LIMIT = 8 * 1024 * REPART_MB_SECTORS,

	REPART_APM_MINIMUM_PARTITIONS = 8,
	REPART_APM_MAXIMUM_PARTITIONS = 63,
};

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
ARC_STATUS ArcFsPartitionObtain(PDEVICE_VECTORS DeviceVectors, ULONG DeviceId, ULONG PartitionId, ULONG SectorSize, PULONG SectorStart, PULONG SectorCount);

/// <summary>
/// Gets the number of partitions (that can be read by this ARC firmware) on a disk.
/// </summary>
/// <param name="DeviceId">Device ID.</param>
/// <param name="PartitionCount">On success obtains the number of partitions</param>
/// <returns>ARC status code</returns>
ARC_STATUS ArcFsPartitionCount(ULONG DeviceId, PULONG PartitionCount);

/// <summary>
/// Gets the number of MBR partitions on a disk.
/// </summary>
/// <param name="DeviceVectors">Device function table.</param>
/// <param name="DeviceId">Device ID.</param>
/// <param name="SectorSize">Sector size for the device.</param>
/// <returns>Number of partitions or 0 on failure</returns>
ULONG ArcFsMbrPartitionCount(PDEVICE_VECTORS DeviceVectors, ULONG DeviceId, ULONG SectorSize);

/// <summary>
/// Repartitions a USB disk, writing an MBR partition table, ARC environment space, an NT partition and an ARC system partition.
/// </summary>
/// <param name="DeviceId">Device ID</param>
/// <param name="NtPartMb">Disk space in MB for the NT partition</param>
/// <param name="DataWritten">If failed, will be set to true after data has been written to disk.</param>
/// <returns>ARC status code.</returns>
ARC_STATUS ArcFsRepartitionDisk(ULONG DeviceId, ULONG NtPartMb, bool* DataWritten);


ARC_STATUS FsInitialiseForDevice(ULONG DeviceId);

ARC_STATUS FsUnmountForDevice(ULONG DeviceId);

void FsInitialiseTable(PARC_FILE_TABLE File);
