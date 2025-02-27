#pragma once
NTSTATUS ExiMapInit(void);
void ExiMapFinalise(void);
PVOID ExiAllocateBufferPage(PULONG AllocatedBufferPage);
void ExiFreeBufferPage(ULONG BufferPage);
void ExiInvalidateDcache(const void* p, ULONG len);