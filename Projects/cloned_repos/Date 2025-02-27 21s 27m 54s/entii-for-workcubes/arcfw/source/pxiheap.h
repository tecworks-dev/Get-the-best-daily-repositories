#pragma once
#include <stdbool.h>
#include "types.h"

bool PxiHeapInit(ULONG PhysAddr, ULONG Size);
PVOID PxiIopAlloc(ULONG Size);
PVOID PxiIopAllocAligned(ULONG Size, ULONG Alignment);
void PxiIopFree(PVOID Ptr);