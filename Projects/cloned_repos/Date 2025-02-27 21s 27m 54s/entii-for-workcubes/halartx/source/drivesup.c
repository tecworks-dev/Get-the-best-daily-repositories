// Disk partitioning related functions.
#include "halp.h"
#include "bugcodes.h"

#include "ntddft.h"
#include "ntdddisk.h"
#include "ntdskreg.h"
#include "stdio.h"
#include "string.h"

ARC_STATUS NvpInitNvram(void);

VOID
IoAssignDriveLetters(
    PLOADER_PARAMETER_BLOCK LoaderBlock,
    PSTRING NtDeviceName,
    OUT PUCHAR NtSystemPath,
    OUT PSTRING NtSystemPathString
    )

{
	
	// By this point, we know io is up and running
	// So attempt to init NVRAM here, if it hasn't been already
	NvpInitNvram();
	
	// Just call the kernel implementation
	HalIoAssignDriveLetters(
		LoaderBlock,
		NtDeviceName,
		NtSystemPath,
		NtSystemPathString
	);
	
}

NTSTATUS
IoReadPartitionTable(
    IN PDEVICE_OBJECT DeviceObject,
    IN ULONG SectorSize,
    IN BOOLEAN ReturnRecognizedPartitions,
    OUT PDRIVE_LAYOUT_INFORMATION *PartitionBuffer
    )

{
    // Just call the kernel implementation
	
	return HalIoReadPartitionTable(
		DeviceObject,
		SectorSize,
		ReturnRecognizedPartitions,
		PartitionBuffer
	);
}

NTSTATUS
IoSetPartitionInformation(
    IN PDEVICE_OBJECT DeviceObject,
    IN ULONG SectorSize,
    IN ULONG PartitionNumber,
    IN ULONG PartitionType
    )

{
	// Just call the kernel implementation
	
	return HalIoSetPartitionInformation(
		DeviceObject,
		SectorSize,
		PartitionNumber,
		PartitionType
	);
	
}

NTSTATUS
IoWritePartitionTable(
    IN PDEVICE_OBJECT DeviceObject,
    IN ULONG SectorSize,
    IN ULONG SectorsPerTrack,
    IN ULONG NumberOfHeads,
    IN PDRIVE_LAYOUT_INFORMATION PartitionBuffer
    )

{
	
	// Just call the kernel implementation
	return HalIoWritePartitionTable(
		DeviceObject,
		SectorSize,
		SectorsPerTrack,
		NumberOfHeads,
		PartitionBuffer
	);
}
