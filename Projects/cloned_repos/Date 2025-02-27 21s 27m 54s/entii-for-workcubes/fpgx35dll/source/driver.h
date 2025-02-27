/******************************Module*Header*******************************\
* Module Name: driver.h
*
* contains prototypes for the frame buffer driver.
*
* Copyright (c) 1992-1995 Microsoft Corporation
\**************************************************************************/
#include "stddef.h"
#include "windows.h"
#include "winddi.h"
#include "devioctl.h"
#include "ntddvdeo.h"
#include "debug.h"
#include "dciddi.h"
#include "runtime.h"

#define RtlCopyMemory memcpy
#define DDI_DRIVER_VERSION_NT3 0x00010002

#define EngAllocMem(a,b,c) LocalAlloc(LMEM_FIXED | LMEM_ZEROINIT, b)
#define EngFreeMem(a) LocalFree(a)
#define EngDeviceIoControl(a,b,c,d,e,f,g) (DeviceIoControl(a,b,c,d,e,f,g,NULL) ? ERROR_SUCCESS : GetLastError())

typedef struct  _PDEV
{
    HANDLE  hDriver;                    // Handle to \Device\Screen
    HDEV    hdevEng;                    // Engine's handle to PDEV
    HSURF   hsurfEng;                   // Engine's handle to surface
    SURFOBJ* psurfBigFb;                 // Pointer to big-endian framebuffer where bIsBigEndian==true
    HSURF   hsurfDouble;                // Handle to framebuffer copy where bIsBigEndian==true
    SURFOBJ* psurfDouble;               // Pointer to framebuffer copy where bIsBigEndian==true
    HPALETTE hpalDefault;               // Handle to the default palette for device.
    PBYTE   pjScreen;                   // This is pointer to base screen address
    ULONG   cxScreen;                   // Visible screen width
    ULONG   cyScreen;                   // Visible screen height
    ULONG   ulMode;                     // Mode the mini-port driver is in.
    LONG    lDeltaScreen;               // Distance from one scan to the next.
    ULONG   cScreenSize;                // size of video memory, including
                                        // offscreen memory.
    PVOID   pOffscreenList;             // linked list of DCI offscreen surfaces.
    FLONG   flRed;                      // For bitfields device, Red Mask
    FLONG   flGreen;                    // For bitfields device, Green Mask
    FLONG   flBlue;                     // For bitfields device, Blue Mask
    ULONG   cPaletteShift;              // number of bits the 8-8-8 palette must
                                        // be shifted by to fit in the hardware
                                        // palette.
    ULONG   ulBitCount;                 // # of bits per pel 8,16,24,32 are only supported.
    POINTL  ptlHotSpot;                 // adjustment for pointer hot spot
    ULONG   cPatterns;                  // Count of bitmap patterns created
    HBITMAP ahbmPat[HS_DDI_MAX];        // Engine handles to standard patterns
    VIDEO_POINTER_CAPABILITIES PointerCapabilities; // HW pointer abilities
    PVIDEO_POINTER_ATTRIBUTES pPointerAttributes; // hardware pointer attributes
    DWORD   cjPointerAttributes;        // Size of buffer allocated
    PALETTEENTRY *pPal;                 // If this is pal managed, this is the pal
    BOOL    fHwCursorActive;            // Are we currently using the hw cursor
    BOOL    bSupportDCI;                // Does the miniport support DCI?
    BOOL    bIsBigEndian;               // Vram is mapped big endian and so needs swaps done
} PDEV, *PPDEV;

DWORD getAvailableModes(HANDLE, PVIDEO_MODE_INFORMATION *, DWORD *);
BOOL bInitPDEV(PPDEV, PDEVMODEW, GDIINFO *, DEVINFO *);
BOOL bInitSURF(PPDEV, BOOL);
BOOL bInitPaletteInfo(PPDEV, DEVINFO *);
BOOL bInitPointer(PPDEV, DEVINFO *);
BOOL bInit256ColorPalette(PPDEV);
VOID vDisablePalette(PPDEV);
VOID vDisableSURF(PPDEV);

#define MAX_CLUT_SIZE (sizeof(VIDEO_CLUT) + (sizeof(ULONG) * 256))

//
// Determines the size of the DriverExtra information in the DEVMODE
// structure passed to and from the display driver.
//

#define DRIVER_EXTRA_SIZE 0

//extern USHORT s_DllName[];
#define DLL_NAME                s_DllName   // Name of the DLL in UNICODE
#define STANDARD_DEBUG_PREFIX   "FRAMEBUF: "  // All debug output is prefixed
#define ALLOC_TAG               'bfDD'        // Four byte tag (characters in
                                              // reverse order) used for memory
                                              // allocations


//
// DCI support
//

#define ROUND_TO_64K(LENGTH)  (((ULONG)(LENGTH) + 0x10000 - 1) & ~(0x10000 - 1))

typedef struct _DCISURF
{
    union {

        DCISURFACEINFO SurfaceInfo;

        DCIOFFSCREEN   OffscreenInfo;

        DCIOVERLAY     OverlayInfo;

    };

    PPDEV ppdev;

    //
    // location of the surface in memory.  Simple version of location
    // information.  Could be changed to rectangles.
    //
    // size is the rounded up size of the surface in offscreen.
    //

    ULONG Offset;
    ULONG Size;

    //
    // info for determining type of surfaces
    //

    struct _DCISURF *prevOffscreen;
    struct _DCISURF *nextOffscreen;

    RECT rclDst;
    RECT rclSrc;
    LPRGNDATA lpClipList;
    BOOLEAN mapped;

} DCISURF, *PDCISURF;



//
// DCI support functions
//

BOOL bEnableDCI(PPDEV ppdev);
ULONG DCICreatePrimarySurface(PPDEV ppdev, ULONG cjIn, PVOID pvIn, ULONG cjOut, PVOID pvOut);


