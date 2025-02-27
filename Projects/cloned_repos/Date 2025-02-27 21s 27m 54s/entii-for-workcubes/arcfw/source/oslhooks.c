#include <stddef.h>
#include <memory.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "arc.h"
#include "arcmem.h"
#include "ppcinst.h"
#include "ppchook.h"
#include "oslhooks.h"
#include "timer.h"
#include "pxi.h"
#include "runtime.h"

typedef ARC_STATUS(*tfpBlSetupForNt)(PVOID LoaderParameterBlock);
static tfpBlSetupForNt orig_BlSetupForNt;

static ARC_STATUS hook_BlSetupForNt(PVOID LoaderParameterBlock) {
    // Do final changes to allow NT kernel init to work
    
    // Call the original function, if it failed, return that failure.
    ARC_STATUS Status = orig_BlSetupForNt(LoaderParameterBlock);
    if (ARC_FAIL(Status)) return Status;

    // On Flipper? nothing needs to be done
    if (s_RuntimePointers[RUNTIME_SYSTEM_TYPE].v == ARTX_SYSTEM_FLIPPER) return Status;

    // Shutdown USB subsystem.
    void UlmsFinalise(void);
    UlmsFinalise();
    void UlkShutdown(void);
    UlkShutdown();
    void UlShutdown(void);
    UlShutdown();

    // Shutdown SDMC driver.
    bool SdmcFinalise(void);
    SdmcFinalise();


    // dolphin is broken and *requires* an IOS restart here
    if (s_RuntimePointers[RUNTIME_IN_EMULATOR].v) {
        // Restart IOS.

        ULONG IosVersion = NativeReadBase32((PVOID)0x60000000, 0x3140);
        if (IosVersion < 3 || IosVersion >= 255) IosVersion = 58;

        IOS_HANDLE hEs;
        LONG Result = PxiIopOpen("/dev/es", IOSOPEN_NONE, &hEs);
        if (Result < 0) return _EFAULT;

        static ULONG xTitleId[2] ARC_ALIGNED(32);
        static ULONG cntviews[2] ARC_ALIGNED(32);
        static UCHAR tikviews[0xd8 * 4] ARC_ALIGNED(32);
        static IOS_IOCTL_VECTOR vectors[3] ARC_ALIGNED(32);

        enum {
            IOCTL_ES_LAUNCH = 0x08,
            IOCTL_ES_GETVIEWCNT = 0x12,
            IOCTL_ES_GETVIEWS = 0x13
        };

        xTitleId[0] = IosVersion;
        xTitleId[1] = 1;
        vectors[0].Pointer = xTitleId;
        vectors[0].Length = sizeof(xTitleId);
        vectors[1].Pointer = cntviews;
        vectors[1].Length = sizeof(ULONG);
        Result = PxiIopIoctlv(hEs, IOCTL_ES_GETVIEWCNT, 1, 1, vectors, 0, 0);
        if (Result < 0) return _EFAULT;
        if (cntviews[1] > 4) return _EFAULT;


        vectors[0].Pointer = xTitleId;
        vectors[0].Length = sizeof(xTitleId);
        vectors[1].Pointer = cntviews;
        vectors[1].Length = sizeof(ULONG);
        vectors[2].Pointer = tikviews;
        vectors[2].Length = 0xd8 * cntviews[1];
        Result = PxiIopIoctlv(hEs, IOCTL_ES_GETVIEWS, 2, 1, vectors, 0, 0);
        if (Result < 0) return _EFAULT;
        NativeWriteBase32((PVOID)0x60000000, 0x3140, 0);
        vectors[0].Pointer = xTitleId;
        vectors[0].Length = sizeof(xTitleId);
        vectors[1].Pointer = tikviews;
        vectors[1].Length = 0xd8;
        Result = PxiIopIoctlvReboot(hEs, IOCTL_ES_LAUNCH, 2, 0, vectors, 0, 0);
        if (Result < 0) return _EFAULT;

        while ((LoadToRegister32(NativeReadBase32((PVOID)0x60000000, 0x3140)) >> 16) == 0) udelay(1000);

        for (ULONG counter = 0; counter <= 400; counter++) {
            udelay(1000);

            if ((MmioReadBase32((PVOID)0x6D000000, 4) & 6) != 0)
                break;
        }
#if 0
        // try opening ES again
        if (s_RuntimePointers[RUNTIME_IN_EMULATOR].v == 0) {
            Result = PxiIopOpen("/dev/es", IOSOPEN_NONE, &hEs);
            if (Result < 0) return _EFAULT;
            PxiIopClose(hEs);

            // delay a bit
            mdelay(1000);
        }
#endif

        data_cache_invalidate((void*)0x80000000, 0x4000);
    }

    return Status;
}


void OslHookInit(PVOID BlOpen, PVOID BlFileTable, PVOID BlSetupForNt, PVOID BlReadSignature) {
    (void)BlOpen;
    (void)BlFileTable;
    (void)BlReadSignature;
    orig_BlSetupForNt = (tfpBlSetupForNt)BlSetupForNt;
    PPCHook_Hook((PVOID*)&orig_BlSetupForNt, hook_BlSetupForNt);
}