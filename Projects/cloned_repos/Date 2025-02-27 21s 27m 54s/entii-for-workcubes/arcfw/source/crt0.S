// Startup code.
// Exception handlers are adapted from OpenBIOS:
/*
 *   Creation Date: <2001/06/16 21:30:18 samuel>
 *   Time-stamp: <2003/04/04 16:32:06 samuel>
 *
 *	<init.S>
 *
 *	Asm glue for ELF images run inside MOL
 *
 *   Copyright (C) 2001, 2002, 2003 Samuel Rydh (samuel@ibrium.se)
 *
 *   This program is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU General Public License
 *   as published by the Free Software Foundation
 *
 */

#define _LANGUAGE_ASSEMBLY
#include "asm.h"

.text

	.extern FwMain
	.extern __executable_start
	.globl _start
_start:
// We are currently in little endian mode,
// in translated mode,
// with interrupts disabled.

// The BATs set up are as follows:
// [DI]BAT0: 0x8000_0000 + 256MB => physical 0x0000_0000 (cached, r+w)
// [DI]BAT1: 0x9000_0000 + 256MB => physical 0x1000_0000 (cached, r+w)
// DBAT2:    0x6000_0000 + 256MB => physical 0x0000_0000 (uncached, r+w)
// DBAT3:    0x7000_0000 + 256MB => physical 0x1000_0000 (uncached, r+w)
// The physical memory of each system is as follows:
// Prod DOL => 24MB at 0
// Dev  DOL => 48MB at 0
// Prod RVL => 24MB at 0, 64MB at 0x1000_0000
// Dev  RVL => 24MB at 0, 128MB at 0x1000_0000
// For Cafe, first 256MB of DDR is mapped by a BAT as well as all other memory:
// Prod Cafe => 32MB at 0, ~3MB at 0x0800_0000, 2GB at 0x1000_0000
// Dev  Cafe => 32MB at 0, ~3MB at 0x0800_0000, 3GB at 0x1000_0000

// r3 is the physical address of our hardware description struct.
// Do not bother touching it right now.
// In fact, save it in a higher register:
	nop
	mr r31, r3

#if 0
// do we get here
	nop
	lis r7, 0x6d80
	nop
	ori r7, r7, 0x00c0
	nop
	mfmsr r6
	nop
	rlwinm r6, r6, 2, 29, 29 // MSR_LE ? 4 : 0
	nop
	or r7, r7, r6 // MSR_LE ? 0d80_00c4 : 0d80_00c0
	nop
	lwz r6, 0(r7)
	nop
	sync
	nop
	ori r6, r6, 0x20
	nop
	stw r6, 0(r7)
	nop
	eieio
	nop
	b .
#endif

	
	// set up a stack:
	// we are at 9MB, use ram before it for stack.
	// this way we can set up 1MB at 8MB as firmware temporary.
	lis r1, __executable_start@h
	ori r1, r1, __executable_start@l
	subi r1, r1, 8
	
	// switch into translated mode and jump to FwMain
	mr r3, r31
	oris r3, r3, 0x8000
#if 0
	lis r5, FwMain@h
	ori r5, r5, FwMain@l
	mtsrr0 r5
	mfmsr r4
	ori r4, r4, MSR_DR|MSR_IR
	mtsrr1 r4
	rfi
#endif
	b FwMain

	.globl ReturnToLoader
ReturnToLoader:
	lis r3, 0x8000
	ori r3, r3, 0x1800
	mtsrr0 r3
	mfmsr r3
	// disable MSR_ILE
	rlwinm r3,r3,0,16,14
	mtmsr r3
	// disable MSR_LE into srr1
	rlwinm r3,r3,0,0,30
	mtsrr1 r3
	rfi
