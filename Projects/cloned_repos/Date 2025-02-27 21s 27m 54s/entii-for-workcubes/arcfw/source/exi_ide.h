#pragma once
#include "types.h"

typedef enum {
	IDE_DRIVE_CARD_A, // Memory Card Slot A (EXI0:0)
	IDE_DRIVE_CARD_B, // Memory Card Slot B (EXI1:0)
	IDE_DRIVE_SP1, // Flipper SP1 (EXI0:2)
	IDE_DRIVE_SP2, // Flipper SP2 (EXI2:0)
} EXI_IDE_DRIVE;

void IdeexiInit(void);
bool IdeexiIsMounted(EXI_IDE_DRIVE drive);
bool IdeexiMount(EXI_IDE_DRIVE drive);
ULONG IdeexiTransferrableSectorCount(EXI_IDE_DRIVE drive);
uint64_t IdeexiSectorCount(EXI_IDE_DRIVE drive);
uint64_t IdeexiReadBlocks(EXI_IDE_DRIVE drive, PVOID buffer, uint64_t sector, ULONG count);
uint64_t IdeexiWriteBlocks(EXI_IDE_DRIVE drive, PVOID buffer, uint64_t sector, ULONG count);
