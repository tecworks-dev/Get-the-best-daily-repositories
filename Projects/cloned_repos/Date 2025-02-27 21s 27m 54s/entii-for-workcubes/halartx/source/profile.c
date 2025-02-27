// Profiler exports.
// We can't implement a profiler due to not having another timing source
// (Vegas timer is for IOP)

#include "halp.h"

void HalStartProfileInterrupt(KPROFILE_SOURCE ProfileSource) {
}

ULONG HalSetProfileInterval(ULONG Interval) {
	return 0;
}

void HalStopProfileInterrupt(KPROFILE_SOURCE ProfileSource) {
}

void HalCalibratePerformanceCounter(volatile PLONG Number) {
	// Raise IRQL to HIGH_LEVEL
	KSPIN_LOCK Lock;
	KIRQL OldIrql;
	
	KeInitializeSpinLock(&Lock);
	KeRaiseIrql(HIGH_LEVEL, &OldIrql);
		
	// Decrement the number of processors, and wait until that same number is zero.
	if (ExInterlockedDecrementLong(Number, &Lock) != RESULT_ZERO) {
		while (*Number != 0) { }
	}
	
	// Zero the performance counters.
	HalpZeroPerformanceCounter();
	
	// Restore IRQL.
	KeLowerIrql(OldIrql);
}