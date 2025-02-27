// Start other processors.
// MPNOTE: this needs implementing.

#include "halp.h"

extern IDTUsage        HalpIDTUsage[MAXIMUM_IDTVECTOR];

static const USHORT s_HalName[] = {
	'A', 'r', 't', 'X', ' ', 'H', 'A', 'L', 0
};

static const UNICODE_STRING s_UsHalName = {
	(sizeof(s_HalName) / sizeof(*s_HalName)) - 1,
	(sizeof(s_HalName) / sizeof(*s_HalName)),
	(USHORT*)s_HalName
};

// Start the next processor.
BOOLEAN HalStartNextProcessor(IN PLOADER_PARAMETER_BLOCK LoaderBlock, IN PKPROCESSOR_STATE ProcessorState) {
	// This is a uniprocessor system.
	return FALSE;
}

// Determine if all processors are started.
BOOLEAN HalAllProcessorsStarted(void) {
	// This is a uniprocessor system.
	return TRUE;
}

// Sends an interprocessor interrupt on a set of processors.
void HalRequestIpi(ULONG Mask) {
	// In Cafe OS:
	// Core2 = bit18, Core1=bit19, Core0=bit20 of SCR
	// for UP, do nothing of course
}

// Handles an interprocessor interrupt: clear the IPI and call the kernel's handler
BOOLEAN HalpHandleIpiInterrupt(IN PKINTERRUPT Interrupt, IN PVOID ServiceContext, IN PVOID TrapFrame) {
	// again: CafeOS unsets the bits mentioned above in SCR.
	KeIpiInterrupt(TrapFrame);
	return TRUE;
}

// Sort partial resource descriptors.
static void HalpGetResourceSortValue (
    IN PCM_PARTIAL_RESOURCE_DESCRIPTOR  pRCurLoc,
    OUT PULONG                          sortscale,
    OUT PLARGE_INTEGER                  sortvalue
    )
{
    switch (pRCurLoc->Type) {
        case CmResourceTypeInterrupt:
            *sortscale = 0;
            sortvalue->LowPart = pRCurLoc->u.Interrupt.Level;
            sortvalue->HighPart = 0;
            break;

        case CmResourceTypePort:
            *sortscale = 1;
            *sortvalue = pRCurLoc->u.Port.Start;
            break;

        case CmResourceTypeMemory:
            *sortscale = 2;
            *sortvalue = pRCurLoc->u.Memory.Start;
            break;

        default:
            *sortscale = 4;
            sortvalue->LowPart = 0;
            sortvalue->HighPart = 0;
            break;
    }
}

void HalReportResourceUsage(void) {
	// Allocate and zero the resource lists
	PCM_RESOURCE_LIST RawResourceList = (PCM_RESOURCE_LIST)ExAllocatePool(NonPagedPool, PAGE_SIZE * 2);
	PCM_RESOURCE_LIST TranslatedResourceList = (PCM_RESOURCE_LIST)ExAllocatePool(NonPagedPool, PAGE_SIZE * 2);
	RtlZeroMemory(RawResourceList, PAGE_SIZE * 2);
	RtlZeroMemory(TranslatedResourceList, PAGE_SIZE * 2);
	
	// Initialise the lists.
	RawResourceList->List[0].InterfaceType = (INTERFACE_TYPE)-1;
	PCM_FULL_RESOURCE_DESCRIPTOR RawFullDesc = RawResourceList->List;
	PCM_FULL_RESOURCE_DESCRIPTOR TlFullDesc = NULL;
	PCM_PARTIAL_RESOURCE_DESCRIPTOR RawThis = (PCM_PARTIAL_RESOURCE_DESCRIPTOR)RawFullDesc;
	PCM_PARTIAL_RESOURCE_DESCRIPTOR TlThis = (PCM_PARTIAL_RESOURCE_DESCRIPTOR)TranslatedResourceList->List;
	PCM_PARTIAL_RESOURCE_LIST RawPartList = &RawFullDesc->PartialResourceList;
	PCM_PARTIAL_RESOURCE_LIST TlPartList = NULL;
	
	for (ULONG i = 0; i < DEVICE_VECTORS; i++) {
		if ((HalpIDTUsage[i].Flags & IDTOwned) != 0) continue;
		HalpIDTUsage[i].Flags = InternalUsage;
		HalpIDTUsage[i].BusReleativeVector = (UCHAR)i;
	}
	
	CM_PARTIAL_RESOURCE_DESCRIPTOR RawDescPart;
	CM_PARTIAL_RESOURCE_DESCRIPTOR TlDescPart;
	
	for (UCHAR pass = 0; pass < 2; pass++) {
		UCHAR ReportOn = (pass == 0 ? DeviceUsage & ~IDTOwned : InternalUsage & ~IDTOwned);
		INTERFACE_TYPE Type = Internal;
		
		ULONG CurrentIDT = 0;
		ULONG CurrentElement = 0;
		while (TRUE) {
			if (CurrentIDT <= MAXIMUM_IDTVECTOR) {
				if ((HalpIDTUsage[CurrentIDT].Flags & ReportOn) == 0) {
					// Doesn't need reporting
					CurrentIDT++;
					continue;
				}
				
				// Report CurrentIDT
				RawDescPart.Type = CmResourceTypeInterrupt;
				RawDescPart.ShareDisposition = CmResourceShareDriverExclusive;
				RawDescPart.Flags = (
					HalpIDTUsage[CurrentIDT].Flags & InterruptLatched ?
					CM_RESOURCE_INTERRUPT_LATCHED :
					CM_RESOURCE_INTERRUPT_LEVEL_SENSITIVE
				);
				RawDescPart.u.Interrupt.Vector = HalpIDTUsage[CurrentIDT].BusReleativeVector;
				RawDescPart.u.Interrupt.Level = RawDescPart.u.Interrupt.Vector;
				RawDescPart.u.Interrupt.Affinity = 0;
				RtlCopyMemory(&TlDescPart, &RawDescPart, sizeof(TlDescPart));
				TlDescPart.u.Interrupt.Vector = CurrentIDT;
				TlDescPart.u.Interrupt.Level = HalpIDTUsage[CurrentIDT].Irql;
				CurrentIDT++;
			} else {
				// TODO: report MMIO usage?
				break;
			}
			
			// Include it in the list
			if (RawFullDesc->InterfaceType != Type) {
				// Type changed, add another section
				RawResourceList->Count++;
				TranslatedResourceList->Count++;
				RawFullDesc = (PCM_FULL_RESOURCE_DESCRIPTOR)RawThis;
				TlFullDesc = (PCM_FULL_RESOURCE_DESCRIPTOR)TlThis;
				RawFullDesc->InterfaceType = Type;
				TlFullDesc->InterfaceType = Type;
				RawPartList = &RawFullDesc->PartialResourceList;
				TlPartList = &TlFullDesc->PartialResourceList;
				
				// and set the iterators
				RawThis = RawFullDesc->PartialResourceList.PartialDescriptors;
				TlThis = TlFullDesc->PartialResourceList.PartialDescriptors;
			}
			
			// Add the new descriptors to the end of the lists.
			RawPartList->Count++;
			TlPartList->Count++;
			RtlCopyMemory(RawThis, &RawDescPart, sizeof(RawDescPart));
			RtlCopyMemory(TlThis, &TlDescPart, sizeof(TlDescPart));
			RawThis++;
			TlThis++;
		}
	}
	
	ULONG ListSize = ( (ULONG)RawThis - (ULONG)RawResourceList );
	
	// Sort the lists basd on the raw resource values
	RawFullDesc = RawResourceList->List;
	TlFullDesc = TranslatedResourceList->List;
	
	for (ULONG i = 0; i < RawResourceList->Count; i++) {
		RawThis = RawFullDesc->PartialResourceList.PartialDescriptors;
		TlThis = TlFullDesc->PartialResourceList.PartialDescriptors;
		ULONG Count = RawFullDesc->PartialResourceList.Count;
		for (ULONG Part = 0; Part < Count; Part++, RawThis++, TlThis++) {
			ULONG CurScale, SortScale;
			LARGE_INTEGER CurValue, SortValue;
			HalpGetResourceSortValue(RawThis, &CurScale, &CurValue);
			PCM_PARTIAL_RESOURCE_DESCRIPTOR RawSort = RawThis;
			PCM_PARTIAL_RESOURCE_DESCRIPTOR TlSort = TlThis;
			
			for (ULONG Sort = Part + 1; Sort < Count; Sort++, RawSort++, TlSort++) {
				HalpGetResourceSortValue(RawSort, &SortScale, &SortValue);
				if (
					(SortScale < CurScale) ||
					(SortScale == CurScale && RtlLargeIntegerLessThan(SortValue, CurValue))
				) {
					// swap Raw
					RtlCopyMemory(&RawDescPart, RawThis, sizeof(RawDescPart));
					RtlCopyMemory(RawThis, RawSort, sizeof(RawDescPart));
					RtlCopyMemory(RawSort, &RawDescPart, sizeof(RawDescPart));
					// and swap Translated
					RtlCopyMemory(&TlDescPart, TlThis, sizeof(TlDescPart));
					RtlCopyMemory(TlThis, TlSort, sizeof(TlDescPart));
					RtlCopyMemory(TlSort, &TlDescPart, sizeof(TlDescPart));
					// we swapped raw with translated, as such:
					HalpGetResourceSortValue(TlThis, &CurScale, &CurValue);
				}
			}
		}
	}
	
	// Tell the kernel about the HAL's resource usage.
	IoReportHalResourceUsage((PUNICODE_STRING)&s_UsHalName, RawResourceList, TranslatedResourceList, ListSize);
	
	// Free the buffers.
	ExFreePool(RawResourceList);
	ExFreePool(TranslatedResourceList);
}