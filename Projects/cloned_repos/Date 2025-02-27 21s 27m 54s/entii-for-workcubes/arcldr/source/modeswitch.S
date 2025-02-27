#define _LANGUAGE_ASSEMBLY
#include "asm.h"

.text

ConfigBATsForArcImpl:
	// clear all BATs
	li		r0,0
	mtspr	IBAT0U,r0; mtspr	IBAT1U,r0; mtspr	IBAT2U,r0; mtspr	IBAT3U,r0		// IBAT0...3
	mtspr	DBAT0U,r0; mtspr	DBAT1U,r0; mtspr	DBAT2U,r0; mtspr	DBAT3U,r0		// DBAT0...3
#ifdef HW_RVL
	mtspr	IBAT4U,r0; mtspr	IBAT5U,r0; mtspr	IBAT6U,r0; mtspr	IBAT7U,r0		// IBAT4...7
	mtspr	DBAT4U,r0; mtspr	DBAT5U,r0; mtspr	DBAT6U,r0; mtspr	DBAT7U,r0		// DBAT4...7
#endif
	isync
	
	// Invalidate all TLBs
	// Comes from mario kart wii forum - ppc pagetable tutorial
	// Invalidate TLBs:
	li r0,64
	li r3,0
	// Wipe SDR1 here:
	sync
	mtspr 25, r3
	isync
	
	mtctr r0
invalidate_tlb_loop:
	tlbie r3
	addi r3, r3, 0x1000
	bdnz+ invalidate_tlb_loop
after_invalidate_tlb:
	tlbsync
	
	// clear all SRs
	lis		r0,0x8000
	mtsr	0,r0; mtsr	1,r0; mtsr	2,r0; mtsr	3,r0; mtsr	4,r0; mtsr	5,r0; mtsr	6,r0
	mtsr	7,r0; mtsr	8,r0; mtsr	9,r0; mtsr	10,r0; mtsr	11,r0; mtsr	12,r0; mtsr	13,r0
	mtsr	14,r0; mtsr	15,r0
	isync
	
	// set DBAT0 and IBAT0:
	// 0x8000_0000 + 256MB => physical 0x0000_0000 (cached, r+w)
	li		r3,2
	lis		r4,0x8000
	ori		r4,r4,0x1fff
	mtspr	IBAT0L,r3
	mtspr	IBAT0U,r4
	mtspr	DBAT0L,r3
	mtspr	DBAT0U,r4
	isync
	
	// set DBAT1 and IBAT1:
	// 0x9000_0000 + 256MB => physical 0x1000_0000 (cached, r+w)
	lis r3, 0x1000
	ori r3, r3, 0x2
	lis r4, 0x9000
	ori r4, r4, 0x1fff
	mtspr	IBAT1L,r3
	mtspr	IBAT1U,r4
	mtspr	DBAT1L,r3
	mtspr	DBAT1U,r4
	isync
	
	// set DBAT2:
	// 0x6000_0000 + 256MB => physical 0x0000_0000 (uncached, r+w)
	li r3, 0x2a
	lis r4, 0x6000
	ori r4, r4, 0x1fff
	mtspr DBAT2L, r3
	mtspr DBAT2U, r4
	isync

	// set DBAT3:
	// 0x7000_0000 + 256MB => physical 0x1000_0000 (uncached, r+w)
	lis r3, 0x1000
	ori r3, r3, 0x2a
	lis r4, 0x7000
	ori r4, r4, 0x1fff
	mtspr DBAT3L, r3
	mtspr DBAT3U, r4
	isync
	
	// disable upper BATS on wii
#ifdef HW_RVL
	mfspr r3, HID4
	rlwinm r3, r3, 0, 7, 5
	mtspr HID4, r3
#endif
	
	mfmsr	r3
	ori		r3,r3,MSR_DR|MSR_IR
	mtsrr1	r3
	mflr	r3
	oris    r3,r3,0x8000
	mtsrr0	r3
	rfi

.extern __realmode
ConfigBATsForArc:
	mflr r28
	oris r28, r28, 0x8000
	lis r3, ConfigBATsForArcImpl@h
	ori r3, r3, ConfigBATsForArcImpl@l
	bl __realmode
	mtlr r28
	blr

.global ModeSwitchEntry
ModeSwitchEntry: // (ArcFirmEntry Start, PHW_DESCRIPTION HwDesc)
	
// save our arguments
// r3 (le entrypoint) into srr0
	mr r29, r3
// r4 (argument) into r31
	mr r31, r4
	li r28,-1

// Disable interrupts.
	mfmsr r7
	rlwinm r8, r7, 0, 17, 15
	mtmsr r8
	isync
	
// Disable paired singles instructions.
// Disable load/store quantized instructions.
	mfspr r7, HID2
	rlwinm r8, r7, 0, 3, 1
	rlwinm r8, r8, 0, 1, 31
	mtspr HID2, r8
	isync

// All exceptions lead to infinite loop. No exceptions.
	li r0,0x10
	mtctr r0
	lis r7,0xc000
	lis r8, 0x4800 // b .
exception_wipe_loop:
	stw r8, 4(r7)
	addi r7, r7, 0x100
	bdnz+ exception_wipe_loop

// Hopefully we can get away with this:
	bl ConfigBATsForArc

// Set MSR_ILE
	mfmsr r7
	lis r8, 1
	or r7, r7, r8
	mtmsr r7
	isync
	
	

// set up srr1 ready to set MSR_LE, disable MSR_IR|MSR_DR
	ori r7, r7, 1
	//rlwinm r7, r7, 0, 28, 25
	mtsrr1 r7
	oris r29, r29, 0x8000
	mtsrr0 r29

// srr0 already set up
// set the hwdesc arg (vaddr):
	oris r3, r31, 0x8000
	mr r4, r30
// and rfi into le entrypoint
	rfi