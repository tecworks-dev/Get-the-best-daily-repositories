// Stub for an x86 emulator, needed for devices with MCS-86 option ROMs.
// (which Flipper/Vegas/Latte systems have none of)

#include "halp.h"

BOOLEAN
HalCallBios (
    IN ULONG BiosCommand,
    IN OUT PULONG Eax,
    IN OUT PULONG Ebx,
    IN OUT PULONG Ecx,
    IN OUT PULONG Edx,
    IN OUT PULONG Esi,
    IN OUT PULONG Edi,
    IN OUT PULONG Ebp
    ) {
	return FALSE;
}