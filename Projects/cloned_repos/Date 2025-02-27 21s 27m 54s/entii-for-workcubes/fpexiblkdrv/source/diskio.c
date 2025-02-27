/*-----------------------------------------------------------------------*/
/* Low level disk I/O module SKELETON for FatFs     (C)ChaN, 2019        */
/*-----------------------------------------------------------------------*/
/* If a working storage control module is available, it should be        */
/* attached to the FatFs via a glue function rather than modifying it.   */
/* This is an example of glue functions to attach various exsisting      */
/* storage control modules to the FatFs module with a defined API.       */
/*-----------------------------------------------------------------------*/

#include "ff.h"			/* Obtains integer types */
#include "diskio.h"		/* Declarations of disk functions */
#define DEVL 1
#include <ntddk.h>

typedef enum {
	IDE_DRIVE_CARD_A, // Memory Card Slot A (EXI0:0)
	IDE_DRIVE_CARD_B, // Memory Card Slot B (EXI1:0)
	IDE_DRIVE_SP1, // Flipper SP1 (EXI0:2)
	IDE_DRIVE_SP2, // Flipper SP2 (EXI2:0)
} EXI_IDE_DRIVE;

typedef enum {
	SDMC_DRIVE_CARD_A, // Memory Card Slot A (EXI0:0)
	SDMC_DRIVE_CARD_B, // Memory Card Slot B (EXI1:0)
	SDMC_DRIVE_SP1, // Flipper SP1 (EXI0:2)
	SDMC_DRIVE_SP2, // Flipper SP2 (EXI2:0)
} EXI_SDMC_DRIVE;

BOOLEAN SdmcexiIsMounted(EXI_SDMC_DRIVE drive);
BOOLEAN SdmcexiWriteProtected(EXI_SDMC_DRIVE drive);
ULONG SdmcexiSectorCount(EXI_SDMC_DRIVE drive);
ULONG SdmcexiReadBlocks(EXI_SDMC_DRIVE drive, PVOID buffer, ULONG sector, ULONG count);
ULONG SdmcexiWriteBlocks(EXI_SDMC_DRIVE drive, PVOID buffer, ULONG sector, ULONG count);
BOOLEAN IdeexiIsMounted(EXI_IDE_DRIVE drive);
ULONG IdeexiTransferrableSectorCount(EXI_IDE_DRIVE drive);
unsigned long long IdeexiSectorCount(EXI_IDE_DRIVE drive);
unsigned long long IdeexiReadBlocks(EXI_IDE_DRIVE drive, PVOID buffer, unsigned long long sector, ULONG count);
unsigned long long IdeexiWriteBlocks(EXI_IDE_DRIVE drive, PVOID buffer, unsigned long long sector, ULONG count);



/*-----------------------------------------------------------------------*/
/* Get Drive Status                                                      */
/*-----------------------------------------------------------------------*/

DSTATUS disk_status (
	BYTE pdrv		/* Physical drive nmuber to identify the drive */
)
{
	DSTATUS stat;
	int result;

	switch (pdrv) {
	case DEV_EXI_CARD_A:
	case DEV_EXI_CARD_B:
	case DEV_EXI_SP1:
	case DEV_EXI_SP2:
		if (!SdmcexiIsMounted((EXI_SDMC_DRIVE)pdrv)) {
			if (!IdeexiIsMounted((EXI_IDE_DRIVE)pdrv)) return STA_NOINIT;
			return 0;
		}

		if (SdmcexiWriteProtected((EXI_SDMC_DRIVE)pdrv)) return STA_PROTECT;

		return 0;
	}
	return STA_NOINIT;
}



/*-----------------------------------------------------------------------*/
/* Inidialize a Drive                                                    */
/*-----------------------------------------------------------------------*/

DSTATUS disk_initialize (
	BYTE pdrv				/* Physical drive nmuber to identify the drive */
)
{
	DSTATUS stat;
	int result;

	switch (pdrv) {
	case DEV_EXI_CARD_A:
	case DEV_EXI_CARD_B:
	case DEV_EXI_SP1:
	case DEV_EXI_SP2:
		if (!SdmcexiIsMounted((EXI_SDMC_DRIVE)pdrv)) {
			if (!IdeexiIsMounted((EXI_IDE_DRIVE)pdrv)) return STA_NOINIT;
			return 0;
		}

		return 0;
	}
	return STA_NOINIT;
}



/*-----------------------------------------------------------------------*/
/* Read Sector(s)                                                        */
/*-----------------------------------------------------------------------*/

DRESULT disk_read (
	BYTE pdrv,		/* Physical drive nmuber to identify the drive */
	BYTE *buff,		/* Data buffer to store read data */
	LBA_t sector,	/* Start sector in LBA */
	UINT count		/* Number of sectors to read */
)
{
	DRESULT res;
	int result;

	if (disk_status(pdrv) & (STA_NODISK | STA_NOINIT)) return RES_NOTRDY;

	switch (pdrv) {
	case DEV_EXI_CARD_A:
	case DEV_EXI_CARD_B:
	case DEV_EXI_SP1:
	case DEV_EXI_SP2:
		if (!SdmcexiIsMounted((EXI_SDMC_DRIVE)pdrv)) {
			ULONG sectors = IdeexiReadBlocks((EXI_IDE_DRIVE)pdrv, buff, sector, count);
			if (sectors < count) return RES_ERROR;
			return RES_OK;
		} else {
			ULONG sectors = SdmcexiReadBlocks((EXI_SDMC_DRIVE)pdrv, buff, sector, count);
			if (sectors < count) return RES_ERROR;
			return RES_OK;
		}

		return RES_ERROR;
	}

	return RES_PARERR;
}



/*-----------------------------------------------------------------------*/
/* Write Sector(s)                                                       */
/*-----------------------------------------------------------------------*/

#if FF_FS_READONLY == 0

DRESULT disk_write (
	BYTE pdrv,			/* Physical drive nmuber to identify the drive */
	const BYTE *buff,	/* Data to be written */
	LBA_t sector,		/* Start sector in LBA */
	UINT count			/* Number of sectors to write */
)
{
	DRESULT res;
	int result;

	if (disk_status(pdrv) & (STA_NODISK | STA_NOINIT)) return RES_NOTRDY;

	switch (pdrv) {
	case DEV_EXI_CARD_A:
	case DEV_EXI_CARD_B:
	case DEV_EXI_SP1:
	case DEV_EXI_SP2:
		if (!SdmcexiIsMounted((EXI_SDMC_DRIVE)pdrv)) {
			ULONG sectors = IdeexiWriteBlocks((EXI_IDE_DRIVE)pdrv, buff, sector, count);
			if (sectors < count) return RES_ERROR;
			return RES_OK;
		}
		else {
			ULONG sectors = SdmcexiWriteBlocks((EXI_SDMC_DRIVE)pdrv, buff, sector, count);
			if (sectors < count) return RES_ERROR;
			return RES_OK;
		}

		return RES_ERROR;
	}

	return RES_PARERR;
}

#endif


/*-----------------------------------------------------------------------*/
/* Miscellaneous Functions                                               */
/*-----------------------------------------------------------------------*/

DRESULT disk_ioctl (
	BYTE pdrv,		/* Physical drive nmuber (0..) */
	BYTE cmd,		/* Control code */
	void *buff		/* Buffer to send/receive control data */
)
{
	DRESULT res;
	int result;

	if (disk_status(pdrv) & (STA_NODISK | STA_NOINIT)) return RES_NOTRDY;

	switch (cmd) {
	case CTRL_SYNC:
		return RES_OK;
	case GET_BLOCK_SIZE:
		*(PULONG)buff = 1;
		return RES_OK;
	case GET_SECTOR_COUNT:
		switch (pdrv) {
		case DEV_EXI_CARD_A:
		case DEV_EXI_CARD_B:
		case DEV_EXI_SP1:
		case DEV_EXI_SP2:
			if (!SdmcexiIsMounted((EXI_SDMC_DRIVE)pdrv)) {
				uint64_t count = IdeexiSectorCount((EXI_IDE_DRIVE)pdrv);
				if (count == 0) return RES_ERROR;
				*(LBA_t*)buff = count;
				return RES_OK;
			}
			else {
				uint64_t count = SdmcexiSectorCount((EXI_SDMC_DRIVE)pdrv);
				if (count == 0) return RES_ERROR;
				*(LBA_t*)buff = count;
				return RES_OK;
			}
			return RES_ERROR;
		}
		break;
	case 100: // get maximum transferrable sector count at one time
		switch (pdrv) {
		case DEV_EXI_CARD_A:
		case DEV_EXI_CARD_B:
		case DEV_EXI_SP1:
		case DEV_EXI_SP2:
			if (!SdmcexiIsMounted((EXI_SDMC_DRIVE)pdrv)) {
				uint64_t count = IdeexiTransferrableSectorCount((EXI_IDE_DRIVE)pdrv);
				if (count == 0) return RES_ERROR;
				*(ULONG*)buff = count;
				return RES_OK;
			}
			else {
				*(ULONG*)buff = 0x1000 / 0x200; // card will keep going until it is stopped, just use the same as IOS_SDMC for now.
				return RES_OK;
			}
			return RES_ERROR;
		}
		break;
	}

	return RES_PARERR;
}

