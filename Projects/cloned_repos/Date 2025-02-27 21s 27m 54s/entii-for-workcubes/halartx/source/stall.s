// Timer SPR stuff

#include "kxppc.h"
#include "halasm.h"

        .extern HalpPerformanceFrequency

// Registers
// KeStallExecutionProcessor argument
        .set    Microsecs,      r.3

// READ_TB() reads the timebase registers to this pair of registers
        .set    TimerLow32,     r.4
        .set    TimerHigh32,    r.5
// KeStallExecutionProcessor stores the timebase compare value to this pair of registers
        .set    EndTimerLow32,  r.6
        .set    EndTimerHigh32, r.7
// Two scratch registers used across this entire file
        .set    Scratch,        r.8
        .set    Scratch2,       r.9

// KeQueryPerformanceCounter arguments
        .set    RetPtr,         r.3
        .set    Freq,           r.4

// Timebase special purpose registers, used in HalpZeroPerformanceCounter
        .set    TBLW,             284
        .set    TBUW,            285

// Reads the timebase registers. Clobbers scratch register.
#define READ_TB() \
	/* read timer registers */ \
	mftbu	TimerHigh32 ; \
	mftb	TimerLow32 ; \
	/*  ensure TimerHigh32 didn't tick over */ \
	mftbu	Scratch ; \
	cmplw	Scratch, TimerHigh32 ; \
	bne-	$ - (4*4) ; /* the things to do to not specify a label here */

// Puts HalpPerformanceFrequency into scratch register
#define GET_PERF_FREQ() \
	/* read address */ \
	lwz		Scratch, [toc]HalpPerformanceFrequency(r.toc) ; \
	lwz		Scratch, 0(Scratch)


// Stalls execution for at least the specified number of microseconds:
// void KeStallExecutionProcessor(ULONG Microseconds)

        LEAF_ENTRY(KeStallExecutionProcessor)

// if Microsecs == 0 then do nothing
        cmplwi   Microsecs, 0
        beqlr-
	
	READ_TB()
		
	GET_PERF_FREQ()

// Microsecs * PerformanceFrequency
        mullw   EndTimerLow32,Microsecs,Scratch
        mulhwu. EndTimerHigh32,Microsecs,Scratch
        bne     StallDiv64
        
// divided by 1000000 (high bits are zero, so just do 32-bit div)
        LWI(Scratch, 1000000)
        divwu   EndTimerLow32,EndTimerLow32,Scratch
        b       StallAfterDiv
// High bits are non-zero, use the optimisation present in other powerpc HALs
// Basically, instead of expensive 64-bit div,
// shift right by 20 (ie, divide by 0x100000)
// and add (shifted value / 16)
// Sacrifices accuracy for speed (computed value is ~1% higher than expected),
// but this function is specified to stall for "at least" the passed in value
// ("waits for at least the specified usecs count, but not significantly longer")
// and is only meant to be called for small values anyway
// ("you must minimize the stall interval, typically to less than 50 usecs")
// so this is fine.
StallDiv64:
        mr      Scratch, EndTimerHigh32
        srwi    EndTimerLow32, EndTimerLow32, 20
        srwi    EndTimerHigh32, EndTimerHigh32, 20
        insrwi  EndTimerLow32, Scratch, 20, 0
        
        srwi    Scratch2, EndTimerLow32, 4
        srwi    Scratch, EndTimerHigh32, 4
        insrwi  Scratch2, EndTimerHigh32, 4, 0
        
        addc    EndTimerLow32, EndTimerLow32, Scratch2
        adde    EndTimerHigh32, EndTimerHigh32, Scratch

// Add to the read timebase registers
StallAfterDiv:
        addc    EndTimerLow32,EndTimerLow32,TimerLow32
        adde    EndTimerHigh32,EndTimerHigh32,TimerHigh32

// And spin until timebase reaches that value
StallSpinLoop:
        READ_TB()
        cmplw   TimerHigh32, EndTimerHigh32 
        blt-    StallSpinLoop
        bgt+    StallDone
        cmplw   TimerLow32, EndTimerLow32
        blt     StallSpinLoop

StallDone:
        LEAF_EXIT(KeStallExecutionProcessor)

// Supplies a 64-bit realtime counter.
// void KeQueryPerformanceCounter(PULARGE_INTEGER RetPtr, PLARGE_INTEGER PerformanceFrequency)
        LEAF_ENTRY(KeQueryPerformanceCounter)

// If PerformanceFrequency pointer is NULL don't try to write there
        cmplwi   Freq, 0
        beq     QueryWrittenPerfFreq

        GET_PERF_FREQ()
	
	li Scratch2, 0
	stw Scratch, 0(Freq)
	stw Scratch2, 4(Freq)

// Read the timebase registers and write to RetPtr
QueryWrittenPerfFreq:
	READ_TB()
	stw TimerLow32, 0(RetPtr)
	stw TimerHigh32, 4(RetPtr)
	
	LEAF_EXIT(KeQueryPerformanceCounter)

// Zero the timebase registers
        LEAF_ENTRY(HalpZeroPerformanceCounter)
        li      r.3, 0
        mtspr   TBLW, r.3
        mtspr   TBUW, r.3
        LEAF_EXIT(HalpZeroPerformanceCounter)

// Read the timebase registers and return in r3-r4
	LEAF_ENTRY(HalpReadTimeBase)
	READ_TB()
	mr r.3, TimerLow32
	mr r.4, TimerHigh32
	LEAF_EXIT(HalpReadTimeBase)