
#define DEVL 1
#include <ntddk.h>
#include "backport.h"
#include "exi_map.h"

enum {
	EXI_BUFFER_LENGTH = 0x80000
};

static PVOID s_ExiDmaBuffer = NULL;
static KSPIN_LOCK s_ExiSpinLock;
static RTL_BITMAP s_ExiDmaMap;
static ULONG s_ExiDmaMapData[4] = {0};

NTSTATUS ExiMapInit(void) {
	PHYSICAL_ADDRESS HighestAddress = {0};
	HighestAddress.LowPart = HighestAddress.HighPart = 0xFFFFFFFFul;
	s_ExiDmaBuffer = MmAllocateContiguousMemory(EXI_BUFFER_LENGTH, HighestAddress);
	if (s_ExiDmaBuffer == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	// Initialise the bitmap.
	RtlInitializeBitMap(&s_ExiDmaMap, s_ExiDmaMapData, sizeof(s_ExiDmaMap) * 8);
	KeInitializeSpinLock(&s_ExiSpinLock);
	return STATUS_SUCCESS;
}

void ExiMapFinalise(void) {
	MmFreeContiguousMemory(s_ExiDmaBuffer);
}

static PVOID ExipAllocateBufferPage(PULONG AllocatedBufferPage) {
	if (AllocatedBufferPage == NULL) return NULL;
	ULONG BufferPage = RtlFindClearBitsAndSet(&s_ExiDmaMap, 1, 0);
	if (BufferPage == 0xFFFFFFFF) {
		return NULL;
	}
	*AllocatedBufferPage = BufferPage;
	return (PUCHAR)s_ExiDmaBuffer + (BufferPage * PAGE_SIZE);
}

PVOID ExiAllocateBufferPage(PULONG AllocatedBufferPage) {
	KIRQL OldIrql;
	//KeAcquireSpinLock(&s_ExiSpinLock, &OldIrql);
	ULONG BufferPage;
	PVOID Ret = ExipAllocateBufferPage(&BufferPage);
	//KeReleaseSpinLock(&s_ExiSpinLock, OldIrql);
	if (Ret == NULL) return Ret;
	*AllocatedBufferPage = BufferPage;
	return Ret;
}

void ExiFreeBufferPage(ULONG BufferPage) {
	KIRQL OldIrql;
	//KeAcquireSpinLock(&s_ExiSpinLock, &OldIrql);
	RtlClearBits(&s_ExiDmaMap, BufferPage, 1);
	//KeReleaseSpinLock(&s_ExiSpinLock, OldIrql);
}

void ExiInvalidateDcache(const void* p, ULONG len) {
	ULONG a, b;

	a = (ULONG)p & ~0x1f;
	b = ((ULONG)p + len + 0x1f) & ~0x1f;

	for (; a < b; a += 32)
		asm("dcbi 0,%0 ; sync" : : "b"(a));

	asm("sync ; isync");
}