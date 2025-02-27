#pragma once
#include "ios_usb.h"

typedef struct _USBMS_CONTROLLER USBMS_CONTROLLER, * PUSBMS_CONTROLLER;

typedef struct _USBMS_DEVICES {
	ULONG DeviceCount;
	ULONG ArcKey[USB_COUNT_DEVICES];
} USBMS_DEVICES, *PUSBMS_DEVICES;

bool UlmsInit(void);
void UlmsFinalise(void);
void UlmsGetDevices(PUSBMS_DEVICES Devices);
PUSBMS_CONTROLLER UlmsGetController(ULONG ArcKey);
ULONG UlmsGetLuns(PUSBMS_CONTROLLER Controller);
ULONG UlmsGetSectorSize(PUSBMS_CONTROLLER Controller, ULONG Lun);
ULONG UlmsGetSectorCount(PUSBMS_CONTROLLER Controller, ULONG Lun);
ULONG UlmsReadSectors(PUSBMS_CONTROLLER Controller, ULONG Lun, ULONG Sector, ULONG NumSector, PVOID Buffer);
ULONG UlmsWriteSectors(PUSBMS_CONTROLLER Controller, ULONG Lun, ULONG Sector, ULONG NumSector, const void* Buffer);
