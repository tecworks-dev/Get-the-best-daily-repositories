#pragma once
#include "types.h"

typedef enum {
	SDMC_DRIVE_CARD_A, // Memory Card Slot A (EXI0:0)
	SDMC_DRIVE_CARD_B, // Memory Card Slot B (EXI1:0)
	SDMC_DRIVE_SP1, // Flipper SP1 (EXI0:2)
	SDMC_DRIVE_SP2, // Flipper SP2 (EXI2:0)
} EXI_SDMC_DRIVE;

void SdmcexiInit(void);
bool SdmcexiIsMounted(EXI_SDMC_DRIVE drive);
bool SdmcexiMount(EXI_SDMC_DRIVE drive);
bool SdmcexiWriteProtected(EXI_SDMC_DRIVE drive);
ULONG SdmcexiSectorCount(EXI_SDMC_DRIVE drive);
ULONG SdmcexiReadBlocks(EXI_SDMC_DRIVE drive, PVOID buffer, ULONG sector, ULONG count);
ULONG SdmcexiWriteBlocks(EXI_SDMC_DRIVE drive, PVOID buffer, ULONG sector, ULONG count);
