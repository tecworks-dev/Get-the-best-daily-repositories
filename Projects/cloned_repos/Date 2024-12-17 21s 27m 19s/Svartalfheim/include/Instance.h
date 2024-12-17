#pragma once

#include <windows.h>
#include <winhttp.h>

#include "Macros.h"

typedef struct _SYS_INFO {
    void*   pAddress;
    WORD    syscall;
} SYS_INFO, *PSYS_INFO;


typedef struct _INSTANCE
{
    struct
    {
        void* pStartAddr;
        void* pEnd;
        DWORD dwSize;
    } Info;

    struct
    {
        void *Kernel32;
        void *Kernelbase;
        void *Ntdll;
        void *Winhttp;
    } Module;

    struct
    {
        D_API(LocalAlloc);
        D_API(LocalReAlloc);
        D_API(LocalFree);

        void* LoadLibraryA;

    } Win32;

    struct
    {
        void* WinHttpOpen;
        void* WinHttpConnect;
        void* WinHttpOpenRequest;
        void* WinHttpSendRequest;
        void* WinHttpReceiveResponse;
        void* WinHttpReadData;
        void* WinHttpCloseHandle;
    } Transport;

    struct
    {
        SYS_INFO NtAllocateVirtualMemory;
        SYS_INFO NtProtectVirtualMemory;
    } Sys;

} INSTANCE, *PINSTANCE;