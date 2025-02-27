// Function called where nothing else needs to be done until next interrupt.
#include "kxppc.h"
#include "halasm.h"

	.set	HID0, 1008

	LEAF_ENTRY(HalProcessorIdle)

	mfmsr r.4
	ori r.4, r.4, 0x8000 // enable interrupts (MSR[EE])
	
	mfspr r.3, HID0
	ori r.3, r.3, 0x0080 // enable doze mode
	mtspr HID0, r.3
	
	oris r.4, r.4, 0x0004 // enable power management (MSR[POW])
	sync
	mtmsr r.4
	isync
	
	LEAF_EXIT(HalpProcessorIdle)
