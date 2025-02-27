// HAL initialisation functions.
#include "halp.h"

ULONG HalpInitPhase;

BOOLEAN HalpPxiInit(void);
BOOLEAN HalExiInit(void);

void HalpSetMmioDbat(void);

BOOLEAN HalpFixLowMem(PLOADER_PARAMETER_BLOCK LoaderBlock);


NTHALAPI
BOOLEAN
HalInitSystem (
    IN ULONG Phase,
    IN PLOADER_PARAMETER_BLOCK LoaderBlock
    ) {

	PKPRCB Prcb;
	
	
	// Save off the init phase.
	
	HalpInitPhase = Phase;
	
	// Get the PRCB, which contains the processor number.
	Prcb = PCR->Prcb;
	if ((Phase == 0) || (Prcb->Number != 0)) {
		
		// Phase 0 initialization for first CPU.
		// CPU-specific initialisation for other CPUs.
		// (This gets executed on all processors) 
		
		// Ensure our PRCB structure is aligned with the NT kernel's.
		
		if (Prcb->MajorVersion != PRCB_MAJOR_VERSION) {
			if (Phase == 0 && Prcb->Number == 0) {
				// This is phase 0 on first CPU, attempt to initialise the display for following bugcheck.
				HalpInitializeDisplay0(LoaderBlock);
			}
			KeBugCheck(MISMATCHED_HAL);
		}
		
		// If this is the first CPU, then this is early boot.
		
		if (Prcb->Number == 0) {
			
			// Fix low memory.
			if (!HalpFixLowMem(LoaderBlock)) {
				return FALSE;
			}
			
			// Set the interval clock increment value.
			
			HalpCurrentTimeIncrement = MAXIMUM_INCREMENT;
			HalpNewTimeIncrement =  MAXIMUM_INCREMENT;
			KeSetTimeIncrement(MAXIMUM_INCREMENT, MINIMUM_INCREMENT);
			
			// Initialize all spin locks.
			
			// TODO: MP - actually implement these.
			#if 0
			KeInitializeSpinLock(&HalpDisplayAdapterLock);
			KeInitializeSpinLock(&HalpSystemInterruptLock);
			#endif
			
			// Don't map the EXI regs yet.
			// If a bugcheck occurs they'll get mapped this early.
			
			
			// Initialize the display adapter.
			
			if (!HalpInitializeDisplay0(LoaderBlock)) {
				return FALSE;
			}
		}
		
		// TODO: MP - set this to core id on espresso
		HALPCR->PhysicalProcessor = Prcb->Number;
		
		// Initialise decrementer timer values.
		HalpCalibrateStall();
		
		// Initialise interrupts.
		if (!HalpInitialiseInterrupts()) {
			return FALSE;
		}
		
		// all done
		return TRUE;
		
		
	} else {
		
		if (Phase != 1) return FALSE;
		
		// Phase 1 initialization for first CPU.
		
		//
		// Map I/O space, initialise interrupts
		//
		if (!HalpMapIoSpace()) {
			return FALSE;
		}
		
		if (!HalpCreateSioStructures()) {
			return FALSE;
		}
		
		// Initialise the IOP PXI driver.
		if (!HalpPxiInit()) {
			return FALSE;
		}
		
		// Initialise the EXI driver.
		if (!HalExiInit()) {
			return FALSE;
		}
		
		// atdisk fails to check if it actually got given valid address space, and will null deref => bugcheck.
		// let's just say the "controllers" were already claimed so that atdisk doesn't bugcheck:
		PCONFIGURATION_INFORMATION Config = IoGetConfigurationInformation();
		Config->AtDiskPrimaryAddressClaimed = TRUE;
		Config->AtDiskSecondaryAddressClaimed = TRUE;
		
		// Map MMIO uncached by DBAT1 at 0x8C000000.
		// Dolphin requires the GPU FIFO to be accessed by a BAT;
		// only the actual MEM1 (rounded up to 32MB) is mapped by IBAT0/DBAT0.
		// Therefore, the 0x8C000000 area is unmapped and unused, so, we can use it.
		// Additionally, it is required for EFB to be accessed by a BAT;
		// due to a hardware erratum, a sync instruction must be placed before the load/store instruction;
		// thus any kind of exception after the sync instruction and before the actual load/store will cause hangs/crashes.
		HalpSetMmioDbat();
		
		return TRUE;
		
	}
}

// HalInitializeProcessor is the first hal export called by the kernel.
VOID
HalInitializeProcessor (
    IN ULONG Number
    ) {
	
	// Check the processor type, we expect to be running on type 8.
	// Early Gekkos use type 0x7000, Espresso uses type 0x7001.
	ULONG ProcessorType;
	asm volatile("mfpvr %0\n" : "=r" (ProcessorType));
	ProcessorType >>= 16;
	if (ProcessorType != 8 && ProcessorType != 0x7000 && ProcessorType != 0x7001) {
		// Not running on an Arthur derivative.
		// (Gekko, Broadway and Espresso are Arthur derivatives)
		KeBugCheck(UNSUPPORTED_PROCESSOR);
		__builtin_unreachable();
	}
	
	// This was already done by arcldr.
	#if 0
	// Disable the upper BATs, NT doesn't know about them.
	ULONG HID4 = __mfspr(SPR_HID4);
	HID4 &= ~HID4_SBE;
	__mtspr(SPR_HID4, HID4);
	#endif
	
	// Other HALs initialise the icache/dcache routines here.
	// We know that we are an arthur derivative.
	
	return;
}
