#pragma once
#include "runtime.h"
#include "iosapi.h"

enum {
	USB_IOCTL_GET_VERSION = 0,
	USB_IOCTL_DEVICE_CHANGE,
	USB_IOCTL_SHUTDOWN,
	USB_IOCTL_GET_DEVICE_INFO,
	USB_IOCTL_ATTACH,
	USB_IOCTL_RELEASE,
	USB_IOCTL_ATTACH_FINISH,
	USB_IOCTL_SET_ALTERNATE_SETTING,
	USB_IOCTL_RESET,
	USB_IOCTL_SUSPEND_RESUME = 0x10,
	USB_IOCTL_CANCEL_ENDPOINT,
	USB_IOCTLV_CONTROL_TRANSFER,
	USB_IOCTLV_INTERRUPT_TRANSFER,
	USB_IOCTLV_ISOCHRONOUS_TRANSFER,
	USB_IOCTLV_BULK_TRANSFER
};

enum {
	USB_BUS_OHCI = 0,
	USB_BUS_EHCI = 1,
	USB_BUS_USB1 = USB_BUS_OHCI,
	USB_BUS_USB2 = USB_BUS_EHCI
};

enum {
	USB_MAX_STRING_LENGTH = 0xFF
};

// IOS ioctl buffers.
// For these generic structures, it is required to use native pointers to access.

typedef struct _IOS_USB_PACKET {
	UCHAR Data[32];
} IOS_USB_PACKET, *PIOS_USB_PACKET;

_Static_assert(sizeof(IOS_USB_PACKET) == 0x20);
_Static_assert((sizeof(IOS_USB_PACKET) * 2) == 0x40);

#define IOS_USB_CHECK_LENGTH(Type) _Static_assert(sizeof(Type) == sizeof(IOS_USB_PACKET))
#define IOS_USB_CHECK_LENGTH_V(Type) _Static_assert(sizeof(Type) == (sizeof(IOS_USB_PACKET) * 2))

// GetVersion response
typedef union _IOS_USB_VERSION {
	struct {
		UCHAR Padding;
		UCHAR Major;
		UCHAR Minor;
		UCHAR Revision;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_VERSION, *PIOS_USB_VERSION;
IOS_USB_CHECK_LENGTH(IOS_USB_VERSION);

// GetDeviceChange response
typedef IOS_USB_DEVICE_ENTRY_MAX IOS_USB_DEVICE_CHANGE, *PIOS_USB_DEVICE_CHANGE;
_Static_assert(sizeof(IOS_USB_DEVICE_CHANGE) == 0x180);

// GetDeviceInfo request
typedef union _IOS_USB_GET_DEVICE_INFO_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		UCHAR AlternateSetting;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_GET_DEVICE_INFO_REQ, *PIOS_USB_GET_DEVICE_INFO_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_GET_DEVICE_INFO_REQ);
// GetDeviceInfo response
typedef struct ARC_BE ARC_ALIGNED(4) ARC_PACKED _IOS_USB_GET_DEVICE_INFO_RES_BODY {
	USB_DEVICE Device;
	USB_CONFIGURATION Config;
	USB_INTERFACE Interface;
	USB_ENDPOINT Endpoints[1];
} IOS_USB_GET_DEVICE_INFO_RES_BODY, *PIOS_USB_GET_DEVICE_INFO_RES_BODY;
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Device) == 0x00);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Config) == 0x14);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Interface) == 0x20);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Endpoints[0]) == 0x2C);

typedef struct ARC_BE _IOS_USB_GET_DEVICE_INFO_VEN_RES {
	union ARC_BE {
		struct ARC_BE {
			IOS_USB_HANDLE DeviceHandle;
			ULONG Bus;
		};
		UCHAR _Data[0x14];
	};
	USB_DEVICE Device;
	USB_CONFIGURATION Config;
	USB_INTERFACE Interface;
	USB_ENDPOINT Endpoints[USB_COUNT_ENDPOINTS];
} IOS_USB_GET_DEVICE_INFO_VEN_RES, *PIOS_USB_GET_DEVICE_INFO_VEN_RES;
_Static_assert(sizeof(IOS_USB_GET_DEVICE_INFO_VEN_RES) == 0xC0);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Device) == 0x14);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Config) == 0x28);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Interface) == 0x34);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Endpoints[0]) == 0x40);

typedef struct ARC_BE _IOS_USB_GET_DEVICE_INFO_HID_RES {
	IOS_USB_HANDLE DeviceHandle;
	union ARC_BE {
		struct ARC_BE {
			ULONG Bus;
		};
		IOS_USB_PACKET _Packet;
	};
	USB_DEVICE Device;
	USB_CONFIGURATION Config;
	USB_INTERFACE Interface;
	USB_ENDPOINT Endpoints[USB_COUNT_ENDPOINTS_HID];
} IOS_USB_GET_DEVICE_INFO_HID_RES, *PIOS_USB_GET_DEVICE_INFO_HID_RES;
_Static_assert(sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES) == 0x60);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_HID_RES, Device) == 0x24);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_HID_RES, Config) == 0x38);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_HID_RES, Interface) == 0x44);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_HID_RES, Endpoints[0]) == 0x50);

// Attach request
typedef union _IOS_USB_ATTACH_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_ATTACH_REQ, *PIOS_USB_ATTACH_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_ATTACH_REQ);

// Detach request
typedef IOS_USB_ATTACH_REQ IOS_DETACH_REQ, *PIOS_USB_DETACH_REQ;

// SetAlternateSetting request
typedef union _IOS_USB_SET_ALTERNATE_SETTING_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		USHORT AlternateSetting;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_SET_ALTERNATE_SETTING_REQ, *PIOS_USB_SET_ALTERNATE_SETTING_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_SET_ALTERNATE_SETTING_REQ);

// Reset request
typedef IOS_USB_ATTACH_REQ IOS_USB_RESET_REQ, *PIOS_USB_RESET_REQ;

// SuspendResume request
typedef union _IOS_USB_SUSPEND_RESUME_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG UnusedLong;
		USHORT UnusedShort;
		USHORT Resume;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_SUSPEND_RESUME_REQ, *PIOS_USB_SUSPEND_RESUME_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_SUSPEND_RESUME_REQ);

// CancelEndpoint request
typedef union _IOS_USB_CANCEL_ENDPOINT_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		UCHAR Endpoint;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_CANCEL_ENDPOINT_REQ, *PIOS_USB_CANCEL_ENDPOINT_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_CANCEL_ENDPOINT_REQ);

// ControlTransfer ioctlv inbuf
// vec1 points to the actual buffer and (must be 16-bit) length
typedef union ARC_ALIGNED(8) _IOS_USB_CONTROL_TRANSFER_REQ {
	struct  {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		UCHAR bmRequestType;
		UCHAR bRequest;
		USHORT wValue;
		USHORT wIndex;
	};
	struct  {
		IOS_USB_PACKET _Padding;
		IOS_IOCTL_VECTOR Vectors[2];
	};
	struct  {
		IOS_USB_PACKET __Padding;
		UCHAR __BytePadding[sizeof(IOS_USB_PACKET) - sizeof(PVOID)];
		PVOID Context;
	};
	IOS_USB_PACKET _Packet[2];
} IOS_USB_CONTROL_TRANSFER_REQ, *PIOS_USB_CONTROL_TRANSFER_REQ;
IOS_USB_CHECK_LENGTH_V(IOS_USB_CONTROL_TRANSFER_REQ);
_Static_assert( __builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, bmRequestType) == 8);
_Static_assert( __builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, bRequest) == 9);
_Static_assert( __builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, wValue) == 10);
_Static_assert( __builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, wIndex) == 12);

// InterruptTransfer ioctlv inbuf
// vec1 points to the actual buffer and (must be 16-bit) length
typedef union ARC_ALIGNED(8) _IOS_USB_INTERRUPT_TRANSFER_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		ULONG Write;
		USHORT UnusedShort;
		UCHAR EndpointAddress;
	};
	struct {
		IOS_USB_PACKET _Padding;
		IOS_IOCTL_VECTOR Vectors[2];
	};
	struct {
		IOS_USB_PACKET __Padding;
		UCHAR __BytePadding[sizeof(IOS_USB_PACKET) - sizeof(PVOID)];
		PVOID Context;
	};
	IOS_USB_PACKET _Packet[2];
} IOS_USB_INTERRUPT_TRANSFER_REQ, *PIOS_USB_INTERRUPT_TRANSFER_REQ;
IOS_USB_CHECK_LENGTH_V(IOS_USB_INTERRUPT_TRANSFER_REQ);
_Static_assert( __builtin_offsetof(IOS_USB_INTERRUPT_TRANSFER_REQ, Write) == 8);
_Static_assert( __builtin_offsetof(IOS_USB_INTERRUPT_TRANSFER_REQ, EndpointAddress) == 14);

// IsochronousTransfer ioctlv inbuf
// vec1 is out PU16BE * NumberOfPackets
// vec2 points to the actual buffer and (must be 16-bit) length
typedef union ARC_ALIGNED(8) _IOS_USB_ISOCHRONOUS_TRANSFER_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused[3];
		UCHAR NumberOfPackets;
		UCHAR EndpointAddress;
	};
	struct {
		IOS_USB_PACKET _Padding;
		IOS_IOCTL_VECTOR Vectors[3];
	};
	struct {
		IOS_USB_PACKET __Padding;
		UCHAR __BytePadding[sizeof(IOS_USB_PACKET) - sizeof(PVOID)];
		PVOID Context;
	};
	IOS_USB_PACKET _Packet[2];
} IOS_USB_ISOCHRONOUS_TRANSFER_REQ, *PIOS_USB_ISOCHRONOUS_TRANSFER_REQ;
IOS_USB_CHECK_LENGTH_V(IOS_USB_ISOCHRONOUS_TRANSFER_REQ);
_Static_assert( __builtin_offsetof(IOS_USB_ISOCHRONOUS_TRANSFER_REQ, NumberOfPackets) == 16);
_Static_assert( __builtin_offsetof(IOS_USB_ISOCHRONOUS_TRANSFER_REQ, EndpointAddress) == 17);

// BulkTransfer ioctlv inbuf
// vec1 points to the actual buffer and (must be 16-bit) length
typedef union ARC_ALIGNED(8) _IOS_USB_BULK_TRANSFER_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused[3];
		USHORT UnusedShort;
		UCHAR EndpointAddress;
	};
	struct {
		IOS_USB_PACKET _Padding;
		IOS_IOCTL_VECTOR Vectors[2];
	};
	struct {
		IOS_USB_PACKET __Padding;
		UCHAR __BytePadding[sizeof(IOS_USB_PACKET) - sizeof(PVOID)];
		PVOID Context;
	};
	IOS_USB_PACKET _Packet[2];
} IOS_USB_BULK_TRANSFER_REQ, *PIOS_USB_BULK_TRANSFER_REQ;
IOS_USB_CHECK_LENGTH_V(IOS_USB_BULK_TRANSFER_REQ);
_Static_assert( __builtin_offsetof(IOS_USB_BULK_TRANSFER_REQ, EndpointAddress) == 18);

// Ensure Context pointer is in the same place in all transfer request structures
_Static_assert(
	__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, Context) ==
	__builtin_offsetof(IOS_USB_INTERRUPT_TRANSFER_REQ, Context) &&
	
	__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, Context) ==
	__builtin_offsetof(IOS_USB_ISOCHRONOUS_TRANSFER_REQ, Context) &&
	
	__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, Context) ==
	__builtin_offsetof(IOS_USB_BULK_TRANSFER_REQ, Context)
);