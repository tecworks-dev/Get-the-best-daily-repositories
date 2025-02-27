#pragma once

typedef struct _PXI_HEAP_CHUNK {
	ULONG Status;
	ULONG Size;
	struct _PXI_HEAP_CHUNK
		* Previous, * Next;
} PXI_HEAP_CHUNK, *PPXI_HEAP_CHUNK;

typedef struct _PXI_HEAP {
	PVOID Base;
	ULONG Size;
	PPXI_HEAP_CHUNK FreeList;
	KSPIN_LOCK SpinLock;
	BOOLEAN Initialised;
} PXI_HEAP, *PPXI_HEAP;


BOOLEAN PhCreate(PPXI_HEAP Heap, PVOID Ptr, ULONG Size);
void PhDelete(PPXI_HEAP Heap);
PVOID PhAlloc(PPXI_HEAP Heap, ULONG Size);
PVOID PhAllocAligned(PPXI_HEAP Heap, ULONG Size, ULONG Alignment);
void PhFree(PPXI_HEAP Heap, PVOID Ptr);