#pragma once
#include "types.h"

bool SdmcFinalise(void);
bool SdmcStartup(void);
bool SdmcIsMounted(void);
bool SdmcIsWriteProtected(void);
ULONG SdmcSectorCount(void);
ULONG SdmcReadSectors(ULONG Sector, ULONG NumSector, PVOID Buffer);
ULONG SdmcWriteSectors(ULONG Sector, ULONG NumSector, const void* Buffer);