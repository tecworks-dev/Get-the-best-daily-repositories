#include <stddef.h>
#include <memory.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include "processor.h"
#include "arc.h"
#include "arcterm.h"
#include "arcenv.h"
#include "pxi.h"
#include "runtime.h"

void TermPowerOffSystem(bool Reset) {
	if (Reset) {
		// We are still in ARC firmware and can therefore jump to reset stub if it's present
		if (s_RuntimePointers[RUNTIME_RESET_STUB].v != 0) {
			void ARC_NORETURN ReturnToLoader();
			ReturnToLoader();
		}
	}

	if (s_RuntimePointers[RUNTIME_SYSTEM_TYPE].v == ARTX_SYSTEM_FLIPPER) {
		// Can't shutdown a flipper system from software.
		// Just reset it, even if Reset is false.
		MmioWriteBase16(MEM_PHYSICAL_TO_K1(0x0C000000), 0x2000, 0);
		MmioWriteBase32(MEM_PHYSICAL_TO_K1(0x0C000000), 0x3024, 3);
		MmioWriteBase32(MEM_PHYSICAL_TO_K1(0x0C000000), 0x3024, 0);
		while (1);
	}

	static ULONG __stm_immbufin[0x08] ARC_ALIGNED(32) = { 0,0,0,0,0,0,0,0 };
	static ULONG __stm_immbufout[0x08] ARC_ALIGNED(32) = { 0,0,0,0,0,0,0,0 };
	IOS_HANDLE hStm;
	LONG result = PxiIopOpen("/dev/stm/immediate", IOSOPEN_NONE, &hStm);
	if (result < 0) {
		while (1);
	}
	// try to IOCTL_STM_SHUTDOWN or IOCTL_STM_HOTRESET. don't bother swapping as everything in input is zero, and we shouldn't return.
	// under emulation, always use shutdown, hotreset is currently not implemented.
	memset(&__stm_immbufin, 0, sizeof(__stm_immbufin));
	if (s_RuntimePointers[RUNTIME_IN_EMULATOR].v) Reset = false;
	result = PxiIopIoctl(hStm, Reset ? 0x2001 : 0x2003, __stm_immbufin, sizeof(__stm_immbufin), __stm_immbufout, sizeof(__stm_immbufout), IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	while (1);
}

static void ArcHalt(void) {
	TermPowerOffSystem(true);
}

static void ArcPowerOff(void) {
	TermPowerOffSystem(false);
}

static void ArcRestart(void) {
	TermPowerOffSystem(true);
}

static void ArcReboot(void) {
	TermPowerOffSystem(true);
}

static void ArcEnterInteractiveMode(void) {
	TermPowerOffSystem(true);
}

void ArcTermInit(void) {
	// Initialise the functions implemented here.
	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
	Api->HaltRoutine = ArcHalt;
	Api->PowerDownRoutine = ArcPowerOff;
	Api->RestartRoutine = ArcRestart;
	Api->RebootRoutine = ArcReboot;
	Api->InteractiveModeRoutine = ArcEnterInteractiveMode;
}