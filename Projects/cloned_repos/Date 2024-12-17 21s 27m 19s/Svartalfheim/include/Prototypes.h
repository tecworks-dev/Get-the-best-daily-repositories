#include "windows.h"

#include "Instance.h"

/*  ---------------------
    Api.c 
--------------------- */

PVOID   xGetModuleHandle    (DWORD dwModuleHash);
PVOID   xGetProcAddress     (PVOID pModuleAddr, DWORD dwProcHash);
DWORD   djb2A               (PBYTE str);
DWORD   djb2W               (LPWSTR str);
BOOL    GetSyscall          (PVOID pFunctionAddress, PSYS_INFO sysInfo);
VOID    xMemcpy             (PBYTE dst, PBYTE src, DWORD size);
VOID    xMemset             (PBYTE dst, BYTE c, DWORD size);



/*  ---------------------
    Utils.s 
--------------------- */
PVOID   SpoofStub           (...);

VOID    RegBackup(
    _In_    PVOID   pBackupArray
);

VOID    RegRestore(
    _In_    PVOID   pBackupArray
);

VOID    SysPrepare(
    _In_    WORD    wSyscall,
    _In_    PVOID   pJmpAddr
);

NTSTATUS    SysCall(
    ...
);

/*  ---------------------
    Utils.s 
--------------------- */
PVOID   SpoofRetAddr        (
    _In_    PVOID function, 
    _In_    HANDLE module, 
    _In_    PVOID a, 
    _In_    PVOID b, 
    _In_    PVOID c, 
    _In_    PVOID d, 
    _In_    PVOID e, 
    _In_    PVOID f, 
    _In_    PVOID g, 
    _In_    PVOID h
    );

/*  ---------------------
    Main.c 
--------------------- */
VOID Main(
    _In_    PVOID       Param,
    _In_    PINSTANCE   Inst
);


