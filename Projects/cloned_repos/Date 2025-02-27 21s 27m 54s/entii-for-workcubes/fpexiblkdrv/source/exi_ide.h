#pragma once

typedef enum {
	IDE_DRIVE_CARD_A, // Memory Card Slot A (EXI0:0)
	IDE_DRIVE_CARD_B, // Memory Card Slot B (EXI1:0)
	IDE_DRIVE_SP1, // Flipper SP1 (EXI0:2)
	IDE_DRIVE_SP2, // Flipper SP2 (EXI2:0)
} EXI_IDE_DRIVE;

void IdeexiInit(void);
BOOLEAN IdeexiIsMounted(EXI_IDE_DRIVE drive);
BOOLEAN IdeexiMount(EXI_IDE_DRIVE drive);
ULONG IdeexiTransferrableSectorCount(EXI_IDE_DRIVE drive);
unsigned long long IdeexiSectorCount(EXI_IDE_DRIVE drive);
unsigned long long IdeexiReadBlocks(EXI_IDE_DRIVE drive, PVOID buffer, unsigned long long sector, ULONG count);
unsigned long long IdeexiWriteBlocks(EXI_IDE_DRIVE drive, PVOID buffer, unsigned long long sector, ULONG count);
