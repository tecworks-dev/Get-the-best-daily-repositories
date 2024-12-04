#include <windows.h>
#define SECURITY_WIN32
#include <security.h>
#include <stdio.h>

#define NAM_SAM_COMPAT 2

#include "../../src/psxecute.h"

void start()
{
    void* heap = GetProcessHeap();
    char* fullName = (char*)HeapAlloc(heap, 0x0, 256);
    int   fullNameLen = 256;

    GetUserNameExA(NAM_SAM_COMPAT, fullName, &fullNameLen);
    PSX_PRINT(fullName);
    PSX_PRINT("\n");
}