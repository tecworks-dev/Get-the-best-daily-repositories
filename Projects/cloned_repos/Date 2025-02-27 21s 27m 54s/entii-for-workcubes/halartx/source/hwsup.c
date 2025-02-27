// Hardware support stuff
#include "halp.h"

// Allocate an adapter channel specified by the adapter object.
NTSTATUS
HalAllocateAdapterChannel(
    IN PADAPTER_OBJECT AdapterObject,
    IN PWAIT_CONTEXT_BLOCK Wcb,
    IN ULONG NumberOfMapRegisters,
    IN PDRIVER_CONTROL ExecutionRoutine
    )
{
	// Should not get here.
	return STATUS_INSUFFICIENT_RESOURCES;
}

// Allocates memory for a buffer that can be accessed by the CPU and some other device.
PVOID
HalAllocateCommonBuffer(
    IN PADAPTER_OBJECT AdapterObject,
    IN ULONG Length,
    OUT PPHYSICAL_ADDRESS LogicalAddress,
    IN BOOLEAN CacheEnabled
    )
{
	PVOID virtualAddress;
	PHYSICAL_ADDRESS physicalAddress;

	// Allocate the buffer.
	physicalAddress.LowPart = 0xFFFFFFFF;
	physicalAddress.HighPart = 0;
	
	virtualAddress = MmAllocateContiguousMemory(Length, physicalAddress);
	if (virtualAddress == NULL) return NULL;

	// Convert virtual to physical and return virtaddr.
	*LogicalAddress = MmGetPhysicalAddress(virtualAddress);
	return virtualAddress;
}

// Allocate memory used by the crash dump handler.

PVOID
HalAllocateCrashDumpRegisters(
    IN PADAPTER_OBJECT AdapterObject,
    IN PULONG NumberOfMapRegisters
    )
{
	// Should not get here.
	return NULL;
}

// Flush any hardware adapter buffers when some other device writes to the common buffer.
BOOLEAN
HalFlushCommonBuffer(
    IN PADAPTER_OBJECT AdapterObject,
    IN ULONG Length,
    IN PHYSICAL_ADDRESS LogicalAddress,
    IN PVOID VirtualAddress
    )
{
	return TRUE;
}

// Frees a common buffer.
VOID
HalFreeCommonBuffer(
    IN PADAPTER_OBJECT AdapterObject,
    IN ULONG Length,
    IN PHYSICAL_ADDRESS LogicalAddress,
    IN PVOID VirtualAddress,
    IN BOOLEAN CacheEnabled
    )
{
	UNREFERENCED_PARAMETER( AdapterObject );
	UNREFERENCED_PARAMETER( Length );
	UNREFERENCED_PARAMETER( LogicalAddress );
	UNREFERENCED_PARAMETER( CacheEnabled );

	MmFreeContiguousMemory (VirtualAddress);
}

// Return an adapter object for a device.
PADAPTER_OBJECT
HalGetAdapter(
    IN PDEVICE_DESCRIPTION DeviceDescription,
    IN OUT PULONG NumberOfMapRegisters
    )
{
	// Check the version.
	if (DeviceDescription->Version > DEVICE_DESCRIPTION_VERSION1) {
		return NULL;
	}
	// Unsupported?
	return NULL;
}

// Frees map registers for an adapter.
VOID
IoFreeMapRegisters(
   PADAPTER_OBJECT AdapterObject,
   PVOID MapRegisterBase,
   ULONG NumberOfMapRegisters
   )
{
	// no operation
}

// Frees an adapter object.
VOID
IoFreeAdapterChannel(
    IN PADAPTER_OBJECT AdapterObject
    )
{
	// no operation
}

// Set up registers in a DMA controller to allow a transfer to or from a device
PHYSICAL_ADDRESS
IoMapTransfer(
    IN PADAPTER_OBJECT AdapterObject,
    IN PMDL Mdl,
    IN PVOID MapRegisterBase,
    IN PVOID CurrentVa,
    IN OUT PULONG Length,
    IN BOOLEAN WriteToDevice
    )
{
	// should be able to get away with this?
	return MmGetPhysicalAddress(CurrentVa);
}

// Flushes the DMA adapter object buffers
BOOLEAN
IoFlushAdapterBuffers(
    IN PADAPTER_OBJECT AdapterObject,
    IN PMDL Mdl,
    IN PVOID MapRegisterBase,
    IN PVOID CurrentVa,
    IN ULONG Length,
    IN BOOLEAN WriteToDevice
    )
{
	// unimplemented
	return FALSE;
}

// handles a machine check interrupt
BOOLEAN HalpHandleMachineCheck(PKINTERRUPT Interrupt, PVOID ServiceContext) {
	KIRQL OldIrql;
	// raise irql
	KeRaiseIrql(MACHINE_CHECK_LEVEL, &OldIrql);
	
	// bugcheck
	KeBugCheck(NMI_HARDWARE_FAILURE);
	
	// should never get here but...
	KeLowerIrql(OldIrql);
	return TRUE;
}

// Returns number of bytes left to be DMA'd
ULONG
HalReadDmaCounter(
	IN PADAPTER_OBJECT AdapterObject
	)
{
	// no operation
	return 0;
}