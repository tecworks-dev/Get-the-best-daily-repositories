// Map all needed MMIO after memory manager init.

#include "halp.h"

BOOLEAN HalpMapInterruptRegs(void);
BOOLEAN HalpMapExiRegs(void);
void HalpTermInit(void);
BOOLEAN HalpInitializeDisplay1(void);

BOOLEAN HalpMapIoSpace(void) {
	if (!HalpMapInterruptRegs()) return FALSE;
	if (!HalpMapExiRegs()) return FALSE;
	HalpTermInit();
	if (!HalpInitializeDisplay1()) return FALSE;
	return TRUE;
}