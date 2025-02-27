/*-----------------------------------------------------------------------*/
/* Low level disk I/O module SKELETON for FatFs     (C)ChaN, 2019        */
/*-----------------------------------------------------------------------*/
/* If a working storage control module is available, it should be        */
/* attached to the FatFs via a glue function rather than modifying it.   */
/* This is an example of glue functions to attach various exsisting      */
/* storage control modules to the FatFs module with a defined API.       */
/*-----------------------------------------------------------------------*/

#include <stdio.h>
#include "ff.h"			/* Obtains integer types */
#include "diskio.h"		/* Declarations of disk functions */
#include "../ios_sdmc.h"
#include "../exi_sdmc.h"
#include "../exi_ide.h"

_Static_assert(DEV_EXI_CARD_A == SDMC_DRIVE_CARD_A);
_Static_assert(DEV_EXI_CARD_B == SDMC_DRIVE_CARD_B);
_Static_assert(DEV_EXI_SP1 == SDMC_DRIVE_SP1);
_Static_assert(DEV_EXI_SP2 == SDMC_DRIVE_SP2);
_Static_assert(DEV_EXI_CARD_A == IDE_DRIVE_CARD_A);
_Static_assert(DEV_EXI_CARD_B == IDE_DRIVE_CARD_B);
_Static_assert(DEV_EXI_SP1 == IDE_DRIVE_SP1);
_Static_assert(DEV_EXI_SP2 == IDE_DRIVE_SP2);

// Work areas for each drive
static FATFS s_FsMounts[DEV_IOS_SDMC + 1] = { 0 };

void disk_MountAll(void) {
	for (int i = 0; i < DEV_IOS_SDMC + 1; i++) {
		printf("Mounting drive %d...", i);
		if ((disk_status(i) & STA_NOINIT) != 0) {
			printf("not present\r\n");
			continue;
		}
		char path[4] = { '0' + i, ':', 0, 0 };
		FRESULT fr = f_mount(&s_FsMounts[i], path, 0);
		if (fr == FR_OK) printf("ok\r\n");
		else printf("failed %d\r\n", fr);
	}
}


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

	case DEV_IOS_SDMC:
		if (!SdmcIsMounted()) return STA_NOINIT;
		if (SdmcIsWriteProtected()) return STA_PROTECT;
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

	case DEV_IOS_SDMC:
		if (!SdmcIsMounted()) return STA_NOINIT;
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

	case DEV_IOS_SDMC:
	{
		ULONG sectors = SdmcReadSectors(sector, count, buff);
		if (sectors < count) return RES_ERROR;
		return RES_OK;
	}
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

	case DEV_IOS_SDMC:
	{
		ULONG sectors = SdmcWriteSectors(sector, count, buff);
		if (sectors < count) return RES_ERROR;
		return RES_OK;
	}
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
		
		case DEV_IOS_SDMC:
		{
			ULONG count = SdmcSectorCount();
			if (count == 0) return RES_ERROR;
			*(LBA_t*)buff = count;
			return RES_OK;
		}
		break;
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

		case DEV_IOS_SDMC:
		{
			*(ULONG*)buff = 0x1000 / 0x200; // driver splits up in pages, so we do that too
			return RES_OK;
		}
		break;
		}
		break;
	}

	return RES_PARERR;
}

