// RVL/Cafe USBv5 low-level driver (by IOS IPC)
#include <stdio.h>
#include <memory.h>
#include "arc.h"
#include "types.h"
#include "runtime.h"
#include "pxi.h"
#include "timer.h"
#include "ios_usb.h"
#include "ios_usb_kbd.h"

// IOS ioctl buffers.
// For these generic structures, it is required to use native pointers to access.

typedef struct _IOS_USB_PACKET {
	union {
		UCHAR Data[32];
	};
} IOS_USB_PACKET, *PIOS_USB_PACKET;

_Static_assert(sizeof(IOS_USB_PACKET) == 0x20);
_Static_assert((sizeof(IOS_USB_PACKET) * 2) == 0x40);

#define IOS_USB_CHECK_LENGTH(Type) _Static_assert(sizeof(Type) == sizeof(IOS_USB_PACKET))
#define IOS_USB_CHECK_LENGTH_V(Type) _Static_assert(sizeof(Type) == (sizeof(IOS_USB_PACKET) * 2) && __alignof__(Type) == 8)

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
typedef IOS_USB_DEVICE_ENTRY_MAX IOS_USB_DEVICE_CHANGE, * PIOS_USB_DEVICE_CHANGE;
_Static_assert(sizeof(IOS_USB_DEVICE_CHANGE) == 0x180);

// GetDeviceInfo request
typedef union _IOS_USB_GET_DEVICE_INFO_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		UCHAR AlternateSetting;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_GET_DEVICE_INFO_REQ, * PIOS_USB_GET_DEVICE_INFO_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_GET_DEVICE_INFO_REQ);

// GetDeviceInfo response
typedef struct ARC_BE ARC_ALIGNED(4) ARC_PACKED _IOS_USB_GET_DEVICE_INFO_RES_BODY {
	USB_DEVICE Device;
	USB_CONFIGURATION Config;
	USB_INTERFACE Interface;
	USB_ENDPOINT Endpoints[1];
} IOS_USB_GET_DEVICE_INFO_RES_BODY, * PIOS_USB_GET_DEVICE_INFO_RES_BODY;
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Device) == 0x00);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Config) == 0x14);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Interface) == 0x20);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Endpoints[0]) == 0x2C);

typedef struct ARC_BE _IOS_USB_GET_DEVICE_INFO_VEN_RES {
	union {
		struct {
			IOS_USB_HANDLE DeviceHandle;
			ULONG Bus;
		};
		UCHAR _Data[0x14];
	};
	USB_DEVICE Device;
	USB_CONFIGURATION Config;
	USB_INTERFACE Interface;
	USB_ENDPOINT Endpoints[USB_COUNT_ENDPOINTS];
} IOS_USB_GET_DEVICE_INFO_VEN_RES, * PIOS_USB_GET_DEVICE_INFO_VEN_RES;
_Static_assert(sizeof(IOS_USB_GET_DEVICE_INFO_VEN_RES) == 0xC0);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Device) == 0x14);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Config) == 0x28);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Interface) == 0x34);
_Static_assert(__builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Endpoints[0]) == 0x40);

typedef struct ARC_BE _IOS_USB_GET_DEVICE_INFO_HID_RES {
	IOS_USB_HANDLE DeviceHandle;
	union {
		struct {
			ULONG Bus;
		};
		IOS_USB_PACKET _Packet;
	};
	USB_DEVICE Device;
	USB_CONFIGURATION Config;
	USB_INTERFACE Interface;
	USB_ENDPOINT Endpoints[USB_COUNT_ENDPOINTS_HID];
} IOS_USB_GET_DEVICE_INFO_HID_RES, * PIOS_USB_GET_DEVICE_INFO_HID_RES;
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
} IOS_USB_ATTACH_REQ, * PIOS_USB_ATTACH_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_ATTACH_REQ);

// Detach request
typedef IOS_USB_ATTACH_REQ IOS_DETACH_REQ, * PIOS_USB_DETACH_REQ;

// SetAlternateSetting request
typedef union _IOS_USB_SET_ALTERNATE_SETTING_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		USHORT AlternateSetting;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_SET_ALTERNATE_SETTING_REQ, * PIOS_USB_SET_ALTERNATE_SETTING_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_SET_ALTERNATE_SETTING_REQ);

// Reset request
typedef IOS_USB_ATTACH_REQ IOS_USB_RESET_REQ, * PIOS_USB_RESET_REQ;

// SuspendResume request
typedef union _IOS_USB_SUSPEND_RESUME_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG UnusedLong;
		USHORT UnusedShort;
		USHORT Resume;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_SUSPEND_RESUME_REQ, * PIOS_USB_SUSPEND_RESUME_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_SUSPEND_RESUME_REQ);

// CancelEndpoint request
typedef union _IOS_USB_CANCEL_ENDPOINT_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		UCHAR Endpoint;
	};
	IOS_USB_PACKET _Packet;
} IOS_USB_CANCEL_ENDPOINT_REQ, * PIOS_USB_CANCEL_ENDPOINT_REQ;
IOS_USB_CHECK_LENGTH(IOS_USB_CANCEL_ENDPOINT_REQ);

// ControlTransfer ioctlv inbuf
// vec1 points to the actual buffer and (must be 16-bit) length
typedef union ARC_ALIGNED(8) _IOS_USB_CONTROL_TRANSFER_REQ {
	struct {
		IOS_USB_HANDLE DeviceHandle;
		ULONG Unused;
		UCHAR bmRequestType;
		UCHAR bRequest;
		USHORT wValue;
		USHORT wIndex;
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
} IOS_USB_CONTROL_TRANSFER_REQ, * PIOS_USB_CONTROL_TRANSFER_REQ;
IOS_USB_CHECK_LENGTH_V(IOS_USB_CONTROL_TRANSFER_REQ);
_Static_assert(__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, bmRequestType) == 8);
_Static_assert(__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, bRequest) == 9);
_Static_assert(__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, wValue) == 10);
_Static_assert(__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, wIndex) == 12);

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
} IOS_USB_INTERRUPT_TRANSFER_REQ, * PIOS_USB_INTERRUPT_TRANSFER_REQ;
IOS_USB_CHECK_LENGTH_V(IOS_USB_INTERRUPT_TRANSFER_REQ);
_Static_assert(__builtin_offsetof(IOS_USB_INTERRUPT_TRANSFER_REQ, Write) == 8);
_Static_assert(__builtin_offsetof(IOS_USB_INTERRUPT_TRANSFER_REQ, EndpointAddress) == 14);

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
} IOS_USB_ISOCHRONOUS_TRANSFER_REQ, * PIOS_USB_ISOCHRONOUS_TRANSFER_REQ;
IOS_USB_CHECK_LENGTH_V(IOS_USB_ISOCHRONOUS_TRANSFER_REQ);
_Static_assert(__builtin_offsetof(IOS_USB_ISOCHRONOUS_TRANSFER_REQ, NumberOfPackets) == 16);
_Static_assert(__builtin_offsetof(IOS_USB_ISOCHRONOUS_TRANSFER_REQ, EndpointAddress) == 17);

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
} IOS_USB_BULK_TRANSFER_REQ, * PIOS_USB_BULK_TRANSFER_REQ;
IOS_USB_CHECK_LENGTH_V(IOS_USB_BULK_TRANSFER_REQ);
_Static_assert(__builtin_offsetof(IOS_USB_BULK_TRANSFER_REQ, EndpointAddress) == 18);
_Static_assert(__alignof__(IOS_USB_BULK_TRANSFER_REQ) == 8);

// Ensure Context pointer is in the same place in all transfer request structures
_Static_assert(
	__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, Context) ==
	__builtin_offsetof(IOS_USB_INTERRUPT_TRANSFER_REQ, Context) &&

	__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, Context) ==
	__builtin_offsetof(IOS_USB_ISOCHRONOUS_TRANSFER_REQ, Context) &&

	__builtin_offsetof(IOS_USB_CONTROL_TRANSFER_REQ, Context) ==
	__builtin_offsetof(IOS_USB_BULK_TRANSFER_REQ, Context)
	);

#define ALIGN_TO_ULONG(val) (((ULONG)((val)) + 3) & ~3)

static IOS_HANDLE s_hUsbVen = IOS_HANDLE_INVALID;
static IOS_HANDLE s_hUsbHid = IOS_HANDLE_INVALID;

static PIOS_USB_DEVICE_CHANGE s_AttachedDevicesVen = NULL;
static PIOS_USB_DEVICE_CHANGE s_AttachedDevicesHid = NULL;
static ULONG s_AsyncIndexVen;
static ULONG s_AsyncIndexHid;
static volatile bool s_KnownDevicesVen = false;
static volatile bool s_KnownDevicesHid = false;

typedef struct _USB_OPENED_DEVICE {
	IOS_USB_HANDLE DeviceHandle;
	ULONG RefCount;
} USB_OPENED_DEVICE, * PUSB_OPENED_DEVICE;
static USB_OPENED_DEVICE s_OpenedDevices[USB_COUNT_DEVICES] = { 0 };

typedef enum {
	VARIANT_VEN = 1,
	VARIANT_HID = 2
} USB_INTERNAL_ASYNC_VARIANT;

typedef enum {
	STATE_DEVICE_CHANGE = 1,
	STATE_ATTACH_FINISH = 2
} USB_INTERNAL_ASYNC_STATE;

static PIOS_USB_DEVICE_ENTRY UlpFindDevice(
	PIOS_USB_DEVICE_CHANGE Host,
	IOS_USB_HANDLE DeviceHandle
) {
	if (Host == NULL) return NULL;

	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		IOS_USB_HANDLE Self = NativeReadBase32(MMIO_OFFSET(Host, Entries[i].DeviceHandle));
		if (Self == 0) break;
		if (Self == DeviceHandle) return &Host->Entries[i];
	}

	return NULL;
}

static PIOS_USB_DEVICE_ENTRY UlpFindDeviceVen(IOS_USB_HANDLE DeviceHandle) {
	return UlpFindDevice(s_AttachedDevicesVen, DeviceHandle);
}

static PIOS_USB_DEVICE_ENTRY UlpFindDeviceHid(IOS_USB_HANDLE DeviceHandle) {
	return UlpFindDevice(s_AttachedDevicesHid, DeviceHandle);
}

static IOS_HANDLE UlpGetIosForUsb(IOS_USB_HANDLE DeviceHandle) {
	// BUGBUG: is this still accurate here?
	if (DeviceHandle < 0) return s_hUsbVen;
	return s_hUsbHid;
}

static LONG UlpIoctl(IOS_USB_HANDLE DeviceHandle, ULONG ControlCode, PVOID Input, ULONG LengthInput, PVOID Output, ULONG LengthOutput, bool Async, PVOID Context) {
	IOS_HANDLE Handle = UlpGetIosForUsb(DeviceHandle);
	if (Async) return PxiIopIoctlAsync(Handle, ControlCode, Input, LengthInput, Output, LengthOutput, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE, Context);
	else return PxiIopIoctl(Handle, ControlCode, Input, LengthInput, Output, LengthOutput, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
}

static LONG UlpIoctlv(IOS_USB_HANDLE DeviceHandle, ULONG ControlCode, ULONG NumRead, ULONG NumWritten, PIOS_IOCTL_VECTOR Buffers, bool EndpointIn, bool Async, PVOID Context) {
	IOS_HANDLE Handle = UlpGetIosForUsb(DeviceHandle);
	// for an endpoint out (write), swap in and out
	// for an endpoint in (read), swap out 
	ULONG SwapIn = 0;
	if (!EndpointIn) SwapIn = ARC_BIT(1);
	ULONG SwapOut = ARC_BIT(1);
	if (Buffers[2].Pointer != NULL) {
		// iso message, buffer 1 is array of u16 packet sizes in wrong endianness, swapping will place everything in correct place and correct endianness for IOP
		SwapIn |= ARC_BIT(1);
		if (!EndpointIn) SwapIn |= ARC_BIT(2);
		SwapOut |= ARC_BIT(2);
	}

	if (Async) return PxiIopIoctlvAsync(Handle, ControlCode, NumRead, NumWritten, Buffers, SwapIn, SwapOut, Context);
	else return PxiIopIoctlv(Handle, ControlCode, NumRead, NumWritten, Buffers, SwapIn, SwapOut);
}

static void ZeroMemory32(void* buffer, ULONG length) {
	if ((length & 3) != 0) {
		memset(buffer, 0, length);
		return;
	}
	length /= sizeof(ULONG);
	PULONG buf32 = (PULONG)buffer;
	for (ULONG i = 0; i < length; i++) buf32[i] = 0;
}

static void memcpy32(void* dest, const void* src, ULONG len) {
	PULONG dest32 = (PULONG)dest;
	const ULONG* src32 = (const ULONG*)src;

	if ((len & 3) != 0) return;

	len /= sizeof(ULONG);
	for (ULONG i = 0; i < len; i++) dest32[i] = src32[i];
}

static ULONG UlpSendIsoMessage(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR bEndpoint,
	UCHAR bPackets,
	PU16BE rpPacketSizes, // << leave this as opposite endian, swap will put everything in correct place
	PVOID rpData,
	bool Async
) {
	if (rpPacketSizes == NULL) return -1;
	if (rpData == NULL) return -1;

	USHORT wLength = 0;
	for (ULONG i = 0; i < bPackets; i++) wLength += rpPacketSizes[i].v;
	if (wLength == 0) return -1;

	PIOS_USB_ISOCHRONOUS_TRANSFER_REQ Req =
		(PIOS_USB_ISOCHRONOUS_TRANSFER_REQ)
		PxiIopAlloc(sizeof(IOS_USB_ISOCHRONOUS_TRANSFER_REQ));
	if (Req == NULL) return -1;

	// Work around the requirement of 32-bit writes to uncached DDR mappings.
	// RtlCopyMemory is guaranteed to always use 32-bit writes;
	// as sizeof(Req) is 32 bit aligned.
	IOS_USB_ISOCHRONOUS_TRANSFER_REQ _Req;
	PIOS_USB_ISOCHRONOUS_TRANSFER_REQ pReq = &_Req;
	ZeroMemory32(pReq, sizeof(Req));

	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, NumberOfPackets, bPackets);
	NATIVE_WRITE(pReq, EndpointAddress, bEndpoint);

	pReq->Vectors[0].Pointer = Req;
	pReq->Vectors[0].Length = sizeof(*Req);
	pReq->Vectors[1].Pointer = rpPacketSizes;
	pReq->Vectors[1].Length = sizeof(USHORT) * bPackets;
	pReq->Vectors[2].Pointer = rpData;
	pReq->Vectors[2].Length = wLength;

	memcpy32(Req, pReq, sizeof(*Req));

	bool EndpointIn = (bEndpoint & USB_ENDPOINT_IN) != 0;

	LONG Status = UlpIoctlv(DeviceHandle, USB_IOCTLV_ISOCHRONOUS_TRANSFER, 1, 2, Req->Vectors, EndpointIn, Async, Req);
	if (!Async || Status < 0) PxiIopFree(Req);

	return Status;
}

static LONG UlpSendControlMessage(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR bmRequestType,
	UCHAR bRequest,
	USHORT wValue,
	USHORT wIndex,
	USHORT wLength,
	PVOID rpData,
	bool Async
) {
	if (rpData == NULL && wLength != 0) return -1;
	if (wLength == 0 && rpData != NULL) return -1;

	PIOS_USB_CONTROL_TRANSFER_REQ Req = (PIOS_USB_CONTROL_TRANSFER_REQ)
		PxiIopAlloc(sizeof(IOS_USB_CONTROL_TRANSFER_REQ));
	if (Req == NULL) return -5;

	// Work around the requirement of 32-bit writes to uncached DDR mappings.
	// RtlCopyMemory is guaranteed to always use 32-bit writes;
	// as sizeof(Req) is 32 bit aligned.
	IOS_USB_CONTROL_TRANSFER_REQ _Req;
	PIOS_USB_CONTROL_TRANSFER_REQ pReq = &_Req;

	ZeroMemory32(pReq, sizeof(*Req));

	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, bmRequestType, bmRequestType);
	NATIVE_WRITE(pReq, bRequest, bRequest);
	NATIVE_WRITE(pReq, wValue, wValue);
	NATIVE_WRITE(pReq, wIndex, wIndex);
	pReq->Vectors[0].Pointer = Req;
	pReq->Vectors[0].Length = sizeof(*Req);
	pReq->Vectors[1].Pointer = rpData;
	pReq->Vectors[1].Length = wLength;

	memcpy32(Req, pReq, sizeof(*Req));

	bool EndpointIn = (bmRequestType & USB_CTRLTYPE_DIR_DEVICE2HOST) == 0;

	LONG Status = UlpIoctlv(DeviceHandle, USB_IOCTLV_CONTROL_TRANSFER, 1, 1, Req->Vectors, EndpointIn, Async, Req);
	if (!Async || Status < 0) PxiIopFree(Req);

	return Status;
}

static ULONG UlpSendBulkMessage(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR bEndpoint,
	USHORT wLength,
	PVOID rpData,
	bool Async
) {

	if (rpData == NULL && wLength != 0) return -1;
	if (wLength == 0 && rpData != NULL) return -1;

	PIOS_USB_BULK_TRANSFER_REQ Req = (PIOS_USB_BULK_TRANSFER_REQ)
		PxiIopAlloc(sizeof(IOS_USB_BULK_TRANSFER_REQ));
	if (Req == NULL) return -1;
	// Work around the requirement of 32-bit writes to uncached DDR mappings.
	// RtlCopyMemory is guaranteed to always use 32-bit writes;
	// as sizeof(Req) is 32 bit aligned.
	IOS_USB_BULK_TRANSFER_REQ _Req;
	PIOS_USB_BULK_TRANSFER_REQ pReq = &_Req;

	ZeroMemory32(pReq, sizeof(*Req));

	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, EndpointAddress, bEndpoint);
	pReq->Vectors[0].Pointer = Req;
	pReq->Vectors[0].Length = sizeof(*Req);
	pReq->Vectors[1].Pointer = rpData;
	pReq->Vectors[1].Length = wLength;

	memcpy32(Req, pReq, sizeof(*Req));

	bool EndpointIn = (bEndpoint & USB_ENDPOINT_IN) != 0;

	LONG Status = UlpIoctlv(DeviceHandle, USB_IOCTLV_BULK_TRANSFER, 1, 1, Req->Vectors, EndpointIn, Async, Req);
	if (!Async || Status < 0) PxiIopFree(Req);

	return Status;
}

static LONG UlpSendInterruptMessage(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR bEndpoint,
	USHORT wLength,
	PVOID rpData,
	bool Async
) {

	if (rpData == NULL && wLength != 0) return -1;
	if (wLength == 0 && rpData != NULL) return -1;

	PIOS_USB_INTERRUPT_TRANSFER_REQ Req = (PIOS_USB_INTERRUPT_TRANSFER_REQ)
		PxiIopAlloc(sizeof(IOS_USB_INTERRUPT_TRANSFER_REQ));
	if (Req == NULL) return -1;
	// Work around the requirement of 32-bit writes to uncached DDR mappings.
	// RtlCopyMemory is guaranteed to always use 32-bit writes;
	// as sizeof(Req) is 32 bit aligned.
	IOS_USB_INTERRUPT_TRANSFER_REQ _Req;
	PIOS_USB_INTERRUPT_TRANSFER_REQ pReq = &_Req;

	ZeroMemory32(pReq, sizeof(*Req));

	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, Write, ((bEndpoint & USB_ENDPOINT_IN) == 0));
	NATIVE_WRITE(pReq, EndpointAddress, bEndpoint);
	pReq->Vectors[0].Pointer = Req;
	pReq->Vectors[0].Length = sizeof(*Req);
	pReq->Vectors[1].Pointer = rpData;
	pReq->Vectors[1].Length = wLength;

	memcpy32(Req, pReq, sizeof(*Req));

	bool EndpointIn = (bEndpoint & USB_ENDPOINT_IN) != 0;

	LONG Status = UlpIoctlv(DeviceHandle, USB_IOCTLV_INTERRUPT_TRANSFER, 1, 1, Req->Vectors, EndpointIn, Async, Req);
	if (!Async || Status < 0) PxiIopFree(Req);

	return Status;
}

static inline LONG UlpGetDesc(
	IOS_USB_HANDLE DeviceHandle,
	PVOID Buffer,
	UCHAR ValueHigh,
	UCHAR ValueLow,
	USHORT Index,
	USHORT Size
) {
	UCHAR RequestType = USB_CTRLTYPE_DIR_DEVICE2HOST;
	if (ValueHigh == USB_DT_HID || ValueHigh == USB_DT_REPORT || ValueHigh == USB_DT_PHYSICAL)
		RequestType |= USB_CTRLTYPE_REC_INTERFACE;

	return UlpSendControlMessage(
		DeviceHandle,
		RequestType,
		USB_REQ_GETDESCRIPTOR,
		(ValueHigh << 8) | ValueLow,
		Index,
		Size,
		Buffer,
		false
	);
}

static bool UlpCheckVersionImpl(IOS_HANDLE Handle, PIOS_USB_VERSION UsbVersion) {
	LONG Status = PxiIopIoctl(Handle, USB_IOCTL_GET_VERSION, NULL, 0, UsbVersion, sizeof(*UsbVersion), IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	if (Status < 0) return false;
	if (NativeReadBase8(MMIO_OFFSET(UsbVersion, Major)) != 5) return false;
	if (NativeReadBase8(MMIO_OFFSET(UsbVersion, Minor)) != 0) return false;
	if (NativeReadBase8(MMIO_OFFSET(UsbVersion, Revision)) != 1) return false;
	return true;
}

static bool UlpCheckVersion(IOS_HANDLE Handle) {
	PIOS_USB_VERSION UsbVersion = PxiIopAlloc(sizeof(IOS_USB_VERSION));
	if (UsbVersion == NULL) return false;

	bool Status = UlpCheckVersionImpl(Handle, UsbVersion);
	PxiIopFree(UsbVersion);
	return Status;
}

static LONG UlpSuspendResume(IOS_USB_HANDLE DeviceHandle, bool Resumed) {
	PIOS_USB_SUSPEND_RESUME_REQ Buf = (PIOS_USB_SUSPEND_RESUME_REQ)
		PxiIopAlloc(sizeof(IOS_USB_SUSPEND_RESUME_REQ));
	if (Buf == NULL) return -1;

	IOS_USB_SUSPEND_RESUME_REQ Req;
	PIOS_USB_SUSPEND_RESUME_REQ pReq = &Req;
	ZeroMemory32(&Req, sizeof(Req));
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, Resume, Resumed);
	memcpy32(Buf, &Req, sizeof(Req));
	LONG Status = -1;
	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		Status = PxiIopIoctl(s_hUsbVen, USB_IOCTL_SUSPEND_RESUME, Buf, sizeof(*Buf), NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	}
	else if (UlpFindDeviceHid(DeviceHandle) != NULL) {
		Status = PxiIopIoctl(s_hUsbHid, USB_IOCTL_SUSPEND_RESUME, Buf, sizeof(*Buf), NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	}
	PxiIopFree(Buf);
	return Status;
}

static LONG UlpSuspendDevice(IOS_USB_HANDLE DeviceHandle) {
	return UlpSuspendResume(DeviceHandle, false);
}

static LONG UlpResumeDevice(IOS_USB_HANDLE DeviceHandle) {
	return UlpSuspendResume(DeviceHandle, true);
}

LONG UlInit(void) {
	LONG Status = 0;
	if (s_AttachedDevicesVen == NULL) {
		if (s_hUsbVen == IOS_HANDLE_INVALID) {
			// Open handle.
			LONG VenStatus = PxiIopOpen("/dev/usb/ven", IOSOPEN_NONE, &s_hUsbVen);
			if (VenStatus < 0) {
				return VenStatus;
			}
			// Check version.
			if (!UlpCheckVersion(s_hUsbVen)) {
				// Incorrect version.
				PxiIopClose(s_hUsbVen);
				s_hUsbVen = IOS_HANDLE_INVALID;
				return -1;
			}
		}
		// Allocate memory for attached devices.
		s_AttachedDevicesVen = PxiIopAlloc(sizeof(*s_AttachedDevicesVen));
		if (s_AttachedDevicesVen == NULL) {
			PxiIopClose(s_hUsbVen);
			s_hUsbVen = IOS_HANDLE_INVALID;
			return -1;
		}
		ZeroMemory32(s_AttachedDevicesVen, sizeof(*s_AttachedDevicesVen));
	}

	// Unlike libogc implementation, only get here if usbven is open and is v5.
	if (s_AttachedDevicesHid == NULL) {
		LONG HidStatus;
		if (s_hUsbHid == IOS_HANDLE_INVALID) {
			// Open handle.
			HidStatus = PxiIopOpen("/dev/usb/hid", IOSOPEN_NONE, &s_hUsbHid);
			if (HidStatus >= 0) {
				// Check version.
				if (!UlpCheckVersion(s_hUsbHid)) {
					// Incorrect version.
					PxiIopClose(s_hUsbHid);
					s_hUsbHid = IOS_HANDLE_INVALID;
				}
			}
		}

		if (s_hUsbHid != IOS_HANDLE_INVALID) {
			// Allocate memory for attached devices.
			s_AttachedDevicesHid = PxiIopAlloc(sizeof(*s_AttachedDevicesHid));
			if (s_AttachedDevicesHid == NULL) {
				PxiIopClose(s_hUsbHid);
				PxiIopClose(s_hUsbVen);
				PxiIopFree(s_AttachedDevicesVen);
				s_AttachedDevicesVen = NULL;
				s_hUsbHid = IOS_HANDLE_INVALID;
				s_hUsbVen = IOS_HANDLE_INVALID;
				return -1;
			}
			ZeroMemory32(s_AttachedDevicesHid, sizeof(*s_AttachedDevicesHid));
		}
		else {
			PxiIopClose(s_hUsbVen);
			PxiIopFree(s_AttachedDevicesVen);
			s_AttachedDevicesVen = NULL;
			s_hUsbHid = IOS_HANDLE_INVALID;
			s_hUsbVen = IOS_HANDLE_INVALID;
			return HidStatus;
		}
	}

#if 0
	Status = PsCreateSystemThread(&s_DeviceChangeThread,
		(ACCESS_MASK)0,
		NULL,
		NULL,
		NULL,
		UlpDeviceChangeThread,
		NULL
	);
#endif
	// Start asynchronous DeviceChange ioctls.
	if (Status >= 0) {
		ZeroMemory32(s_OpenedDevices, sizeof(s_OpenedDevices));
		Status = PxiIopIoctlAsync(
			s_hUsbVen,
			USB_IOCTL_DEVICE_CHANGE,
			NULL, 0,
			s_AttachedDevicesVen, sizeof(*s_AttachedDevicesVen),
			IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
			(PVOID)STATE_DEVICE_CHANGE
		);
		//printf("USB: VEN async DeviceChange %d\r\n", Status);
		if (Status >= 0) s_AsyncIndexVen = (ULONG)Status;
	}
	if (Status >= 0) {
		Status = PxiIopIoctlAsync(
			s_hUsbHid,
			USB_IOCTL_DEVICE_CHANGE,
			NULL, 0,
			s_AttachedDevicesHid, sizeof(*s_AttachedDevicesHid),
			IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
			(PVOID)STATE_DEVICE_CHANGE
		);
		//printf("USB: HID async DeviceChange %d\r\n", Status);
		if (Status >= 0) s_AsyncIndexHid = (ULONG)Status;
	}
	if (Status < 0) {
		PxiIopIoctl(s_hUsbHid, USB_IOCTL_SHUTDOWN, NULL, 0, NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
		PxiIopClose(s_hUsbHid);
		PxiIopIoctl(s_hUsbVen, USB_IOCTL_SHUTDOWN, NULL, 0, NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
		PxiIopClose(s_hUsbVen);
		PxiIopFree(s_AttachedDevicesHid);
		s_AttachedDevicesHid = NULL;
		PxiIopFree(s_AttachedDevicesVen);
		s_AttachedDevicesVen = NULL;
		s_hUsbHid = IOS_HANDLE_INVALID;
		s_hUsbVen = IOS_HANDLE_INVALID;
	}
	else {
		// Spin while first DeviceChange is yet to come in for both VEN and HID
		while (!s_KnownDevicesVen || !s_KnownDevicesHid) {
			// poll for the async ops to finish
			UlPoll();
		}
	}

	return Status;
}

void UlShutdown(void) {
	if (s_hUsbVen != IOS_HANDLE_INVALID) {
		PxiIopIoctl(s_hUsbHid, USB_IOCTL_SHUTDOWN, NULL, 0, NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
		PxiIopClose(s_hUsbHid);
		PxiIopIoctl(s_hUsbVen, USB_IOCTL_SHUTDOWN, NULL, 0, NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
		PxiIopClose(s_hUsbVen);
		PxiIopFree(s_AttachedDevicesHid);
		s_AttachedDevicesHid = NULL;
		PxiIopFree(s_AttachedDevicesVen);
		s_AttachedDevicesVen = NULL;
		s_hUsbHid = IOS_HANDLE_INVALID;
		s_hUsbVen = IOS_HANDLE_INVALID;
	}
}

static void UlpAttachFinishCallback(LONG Status, ULONG Result, USB_INTERNAL_ASYNC_VARIANT Variant) {
	PIOS_USB_DEVICE_CHANGE Devices = NULL;
	PULONG AsyncIndex = NULL;
	IOS_HANDLE Handle;
	if (Variant == VARIANT_VEN) {
		Devices = s_AttachedDevicesVen;
		Handle = s_hUsbVen;
		AsyncIndex = &s_AsyncIndexVen;
	}
	else if (Variant == VARIANT_HID) {
		Devices = s_AttachedDevicesHid;
		Handle = s_hUsbHid;
		AsyncIndex = &s_AsyncIndexHid;
	}
	else return; // wtf?

	if (Variant == VARIANT_VEN) s_KnownDevicesVen = true;
	if (Variant == VARIANT_HID) s_KnownDevicesHid = true;

	if (Status < 0) {
		Status = PxiIopIoctlAsync(
			Handle,
			USB_IOCTL_ATTACH_FINISH,
			NULL, 0,
			NULL, 0,
			IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
			(PVOID)STATE_ATTACH_FINISH
		);
		if (Status >= 0) *AsyncIndex = Status;
		return;
	}

	// Call DeviceChange.
	Status = PxiIopIoctlAsync(
		Handle,
		USB_IOCTL_DEVICE_CHANGE,
		NULL, 0,
		Devices, sizeof(*Devices),
		IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
		(PVOID)STATE_DEVICE_CHANGE
	);
	if (Status >= 0) *AsyncIndex = Status;
}

static void UlpDeviceChangeCallback(LONG Status, ULONG Result, USB_INTERNAL_ASYNC_VARIANT Variant) {
	PIOS_USB_DEVICE_CHANGE Devices = NULL;
	PULONG AsyncIndex = NULL;
	ULONG RuntimeIndex = 0;
	IOS_HANDLE Handle;
	if (Variant == VARIANT_VEN) {
		Devices = s_AttachedDevicesVen;
		Handle = s_hUsbVen;
		AsyncIndex = &s_AsyncIndexVen;
	}
	else if (Variant == VARIANT_HID) {
		Devices = s_AttachedDevicesHid;
		Handle = s_hUsbHid;
		AsyncIndex = &s_AsyncIndexHid;
		RuntimeIndex = 1;
	}
	else return; // wtf?

	if (Status < 0) {
		Status = PxiIopIoctlAsync(
			Handle,
			USB_IOCTL_DEVICE_CHANGE,
			NULL, 0,
			Devices, sizeof(*Devices),
			IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
			(PVOID)STATE_DEVICE_CHANGE
		);
		if (Status >= 0) *AsyncIndex = Status;
		return;
	}

	// Zero out unused entries.
	// Do this using NativeWriteBase32 because endianness reasons.
	//printf("USB: got %d devices for %s\r\n", Result, Variant == VARIANT_HID ? "HID" : "VEN");
	ULONG Length = sizeof(Devices->Entries[0]) * (USB_COUNT_DEVICES - Result);
	Length /= sizeof(ULONG);
	for (ULONG i = 0; i < Length; i++) {
		NativeWriteBase32(Devices, __builtin_offsetof(IOS_USB_DEVICE_CHANGE, Entries[Result]) + (i * sizeof(ULONG)), 0);
	}

	// Call AttachFinish.
	Status = PxiIopIoctlAsync(
		Handle,
		USB_IOCTL_ATTACH_FINISH,
		NULL, 0, NULL, 0,
		IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
		(PVOID)STATE_ATTACH_FINISH
	);
	if (Status >= 0) *AsyncIndex = Status;
}

void UlPoll(void) {
	LONG Result;
	PVOID Context;
	USB_INTERNAL_ASYNC_STATE State;
	
	if (s_hUsbHid == IOS_HANDLE_INVALID) return;
	if (s_hUsbVen == IOS_HANDLE_INVALID) return;
	
	udelay(100);

	if (PxiIopIoctlAsyncPoll(s_AsyncIndexVen, &Result, &Context)) {
		//printf("USB: VEN finished %d\r\n", (ULONG)Context);
		State = (USB_INTERNAL_ASYNC_STATE)Context;

		if (State == STATE_DEVICE_CHANGE) UlpDeviceChangeCallback(Result, Result, VARIANT_VEN);
		else if (State == STATE_ATTACH_FINISH) UlpAttachFinishCallback(Result, Result, VARIANT_VEN);
	}

	if (PxiIopIoctlAsyncPoll(s_AsyncIndexHid, &Result, &Context)) {
		//printf("USB: HID finished %d\r\n", (ULONG)Context);
		State = (USB_INTERNAL_ASYNC_STATE)Context;

		if (State == STATE_DEVICE_CHANGE) UlpDeviceChangeCallback(Result, Result, VARIANT_HID);
		else if (State == STATE_ATTACH_FINISH) UlpAttachFinishCallback(Result, Result, VARIANT_HID);
	}

	// TODO: poll other things?
}

static LONG UlpGetDescriptorsHidForOpen(IOS_USB_HANDLE DeviceHandle) {
	PIOS_USB_GET_DEVICE_INFO_REQ Req = (PIOS_USB_GET_DEVICE_INFO_REQ)
		PxiIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_REQ));
	if (Req == NULL) return -1;

	IOS_USB_GET_DEVICE_INFO_REQ _Req;
	PIOS_USB_GET_DEVICE_INFO_REQ pReq = &_Req;
	ZeroMemory32(&_Req, sizeof(_Req));
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, AlternateSetting, 0);
	memcpy32(Req, &_Req, sizeof(_Req));

	PIOS_USB_GET_DEVICE_INFO_HID_RES HidRes = (PIOS_USB_GET_DEVICE_INFO_HID_RES)
		PxiIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES));
	if (HidRes == NULL) {
		PxiIopFree(Req);
		return -1;
	}

	ZeroMemory32(HidRes, sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES));
	LONG Status = PxiIopIoctl(s_hUsbHid, USB_IOCTL_GET_DEVICE_INFO, Req, sizeof(*Req), HidRes, sizeof(*HidRes), IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	PxiIopFree(Req);
	PxiIopFree(HidRes);
	return Status;
}

static LONG UlpClearHaltMsg(IOS_USB_HANDLE DeviceHandle) {
	// some devices can't handle this, so ignore result.
	UlpSendControlMessage(
		DeviceHandle,
		(USB_CTRLTYPE_DIR_HOST2DEVICE | USB_CTRLTYPE_TYPE_STANDARD | USB_CTRLTYPE_REC_ENDPOINT),
		USB_REQ_CLEARFEATURE,
		USB_FEATURE_ENDPOINT_HALT,
		0,
		0,
		NULL,
		false
	);
	return 0;
}

static LONG UlpOpenDevice(IOS_USB_HANDLE DeviceHandle) {
	// for HID, descriptor needs to be read
	if (UlpFindDeviceHid(DeviceHandle) == NULL) {
		if (UlpFindDeviceVen(DeviceHandle) != NULL) {
#if 0
			NTSTATUS Status = UlpResumeDevice(DeviceHandle);
			if (!NT_SUCCESS(Status)) return Status;
			Status = UlpClearHaltMsg(DeviceHandle);
			if (!NT_SUCCESS(Status)) {
				UlpSuspendDevice(DeviceHandle);
			}
			return Status;
#endif
			return UlpResumeDevice(DeviceHandle);
		}
		return -1;
	}
	LONG Status = UlpResumeDevice(DeviceHandle);
	if (Status < 0) return Status;
	//USB_DEVICE_DESC Descriptors;
	Status = UlpGetDescriptorsHidForOpen(DeviceHandle);
	if (Status < 0) {
		UlpSuspendDevice(DeviceHandle);
		return Status;
	}
#if 0
	Status = UlpClearHaltMsg(DeviceHandle);
	if (!NT_SUCCESS(Status)) {
		UlpSuspendDevice(DeviceHandle);
	}
#endif
	return Status;
}

static LONG UlpOpenDeviceEnsureClearHalt(IOS_USB_HANDLE DeviceHandle) {
	return UlpOpenDevice(DeviceHandle);
}

LONG UlOpenDevice(IOS_USB_HANDLE DeviceHandle) {
	PUSB_OPENED_DEVICE EmptyDevice = NULL;
	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		IOS_USB_HANDLE Current = s_OpenedDevices[i].DeviceHandle;
		if (Current == DeviceHandle) {
			s_OpenedDevices[i].RefCount++;
			return 0;
		}
		if (s_OpenedDevices[i].RefCount == 0 && Current == 0 && EmptyDevice == NULL) {
			EmptyDevice = &s_OpenedDevices[i];
		}
	}
	if (EmptyDevice == NULL) return -1;
	LONG Status = UlpOpenDeviceEnsureClearHalt(DeviceHandle);
	if (Status < 0) return Status;
	EmptyDevice->DeviceHandle = DeviceHandle;
	EmptyDevice->RefCount = 1;
	return Status;
}

LONG UlCloseDevice(IOS_USB_HANDLE DeviceHandle) {
	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		IOS_USB_HANDLE Current = s_OpenedDevices[i].DeviceHandle;
		if (Current == DeviceHandle) {
			s_OpenedDevices[i].RefCount--;
			if (s_OpenedDevices[i].RefCount != 0)
				return 0;
			s_OpenedDevices[i].DeviceHandle = 0;
			return UlpSuspendResume(DeviceHandle, false);
		}
	}
	return -1;
}

LONG UlGetDeviceDesc(IOS_USB_HANDLE DeviceHandle, PUSB_DEVICE Device) {
	PUSB_DEVICE Buf = PxiIopAlloc(USB_DT_DEVICE_SIZE);
	if (Buf == NULL) return -1;

	LONG Status = UlpSendControlMessage(DeviceHandle, USB_CTRLTYPE_DIR_DEVICE2HOST, USB_REQ_GETDESCRIPTOR, (USB_DT_DEVICE << 8), 0, USB_DT_DEVICE_SIZE, Buf, false);
	if (Status >= 0) {
		// Copy out "correctly".
#if 0
		NATIVE_COPY_FROM(Device, Buf, bLength);
		NATIVE_COPY_FROM(Device, Buf, bDescriptorType);
		NATIVE_COPY_FROM(Device, Buf, bcdUSB);
		NATIVE_COPY_FROM(Device, Buf, bDeviceClass);
		NATIVE_COPY_FROM(Device, Buf, bDeviceSubClass);
		NATIVE_COPY_FROM(Device, Buf, bDeviceProtocol);
		NATIVE_COPY_FROM(Device, Buf, bMaxPacketSize);
		NATIVE_COPY_FROM(Device, Buf, idVendor);
		NATIVE_COPY_FROM(Device, Buf, idProduct);
		NATIVE_COPY_FROM(Device, Buf, bcdDevice);
		NATIVE_COPY_FROM(Device, Buf, iManufacturer);
		NATIVE_COPY_FROM(Device, Buf, iProduct);
		NATIVE_COPY_FROM(Device, Buf, iSerialNumber);
		NATIVE_COPY_FROM(Device, Buf, bNumConfigurations);
#endif
		memcpy(Device, Buf, sizeof(*Device));
	}
	PxiIopFree(Buf);
	return Status;
}

LONG UlGetDescriptors(IOS_USB_HANDLE DeviceHandle, PUSB_DEVICE_DESC Device) {
	PIOS_USB_DEVICE_ENTRY Ven = UlpFindDeviceVen(DeviceHandle);
	PIOS_USB_DEVICE_ENTRY Hid = UlpFindDeviceHid(DeviceHandle);
	if (Ven == NULL && Hid == NULL) return -1;

	PIOS_USB_GET_DEVICE_INFO_RES_BODY Body = NULL;
	ULONG BodyOffset = 0;

	PIOS_USB_GET_DEVICE_INFO_REQ Req = (PIOS_USB_GET_DEVICE_INFO_REQ)
		PxiIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_REQ));
	if (Req == NULL) return -1;

	{
		IOS_USB_GET_DEVICE_INFO_REQ _Req;
		PIOS_USB_GET_DEVICE_INFO_REQ pReq = &_Req;
		ZeroMemory32(&_Req, sizeof(_Req));
		NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
		NATIVE_WRITE(pReq, AlternateSetting, 0);
		memcpy32(Req, &_Req, sizeof(_Req));
	}
	PVOID Res = NULL;
	ULONG ResLength = 0;
	IOS_HANDLE Handle = IOS_HANDLE_INVALID;
	bool IsHid = false;

	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		PIOS_USB_GET_DEVICE_INFO_VEN_RES VenRes = (PIOS_USB_GET_DEVICE_INFO_VEN_RES)
			PxiIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_VEN_RES));
		if (VenRes == NULL) {
			PxiIopFree(Req);
			return -1;
		}
		Res = VenRes;
		ResLength = sizeof(IOS_USB_GET_DEVICE_INFO_VEN_RES);
		Body = (PIOS_USB_GET_DEVICE_INFO_RES_BODY)&VenRes->Device;
		BodyOffset = __builtin_offsetof(IOS_USB_GET_DEVICE_INFO_VEN_RES, Device);
		Handle = s_hUsbVen;
	}
	else if (UlpFindDeviceHid(DeviceHandle) != NULL) {
		PIOS_USB_GET_DEVICE_INFO_HID_RES HidRes = (PIOS_USB_GET_DEVICE_INFO_HID_RES)
			PxiIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES));
		if (HidRes == NULL) {
			PxiIopFree(Req);
			return -1;
		}
		Res = HidRes;
		ResLength = sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES);
		Body = (PIOS_USB_GET_DEVICE_INFO_RES_BODY)&HidRes->Device;
		BodyOffset = __builtin_offsetof(IOS_USB_GET_DEVICE_INFO_HID_RES, Device);
		Handle = s_hUsbHid;
		IsHid = true;
	}

	ZeroMemory32(Res, ResLength);
	LONG Status = PxiIopIoctl(Handle, USB_IOCTL_GET_DEVICE_INFO, Req, sizeof(*Req), Res, ResLength, IOCTL_SWAP_NONE, IOCTL_SWAP_OUTPUT);

	// Don't need the request buffer any more.
	PxiIopFree(Req);
	Req = NULL;

	if (Status < 0) {
		PxiIopFree(Res);
		return Status;
	}

	// Ensure everything unused is zeroed.
	memset(Device, 0, sizeof(*Device));
	// Unmarshal from bus to PPCLE.
#if 0
	{
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bLength);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bDescriptorType);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bcdUSB);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bDeviceClass);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bDeviceSubClass);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bDeviceProtocol);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bMaxPacketSize);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.idVendor);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.idProduct);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bcdDevice);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.iManufacturer);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.iProduct);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.iSerialNumber);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bNumConfigurations);
	}
#endif

	memcpy(&Device->Device, &Body->Device, sizeof(Device->Device));
	if (Device->Device.bNumConfigurations == 0) return Status;
	if (Device->Device.bNumConfigurations > 1) Device->Device.bNumConfigurations = 1;
#if 0
	{
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bLength);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bDescriptorType);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bTotalLength);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bNumInterfaces);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bConfigurationValue);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.iConfiguration);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bmAttributes);
		NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Device.bMaxPower);
	}
#endif
	memcpy(&Device->Config, &Body->Config, sizeof(Device->Config));

	// IOS claims each interface is its own device=
	if (Device->Config.bNumInterfaces != 0) {
		Device->Config.bNumInterfaces = 1;
#if 0
		{
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.bLength);
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.bDescriptorType);
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.bInterfaceNumber);
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.bAlternateSetting);
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.bNumEndpoints);
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.bInterfaceClass);
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.bInterfaceSubClass);
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.bInterfaceProtocol);
			NATIVE_COPY_FROM_OFFSET(Device, Body, BodyOffset, Interface.iInterface);
		}
#endif
		memcpy(&Device->Interface, &Body->Interface, sizeof(Device->Interface));
		if (IsHid) {
			// HID provides only an interface input and an interface output.
			// If either are not present they will be zeroed.
			UCHAR bNumEndpoints = 0;
			ULONG iEndpoint = 0;
			UCHAR dt = NativeReadBase8(Res, BodyOffset + __builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Endpoints[iEndpoint].bDescriptorType));
			if (dt != 0) {
#if 0
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bLength);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bDescriptorType);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bEndpointAddress);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bmAttributes);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], wMaxPacketSize);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bInterval);
#endif
				memcpy(&Device->Endpoints[bNumEndpoints], &Body->Endpoints[iEndpoint], sizeof(Device->Endpoints[0]));
				bNumEndpoints++;
			}
			iEndpoint++;

			dt = NativeReadBase8(Res, BodyOffset + __builtin_offsetof(IOS_USB_GET_DEVICE_INFO_RES_BODY, Endpoints[iEndpoint].bDescriptorType));
			if (dt != 0) {
#if 0
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bLength);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bDescriptorType);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bEndpointAddress);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bmAttributes);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], wMaxPacketSize);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[bNumEndpoints], Endpoints[iEndpoint], bInterval);
#endif
				memcpy(&Device->Endpoints[bNumEndpoints], &Body->Endpoints[iEndpoint], sizeof(Device->Endpoints[0]));
				bNumEndpoints++;
			}
			Device->Interface.bNumEndpoints = bNumEndpoints;
		}
		else {
			// Skip vendor and class specific descriptors.
			ULONG iEndpoint = 0;
			for (iEndpoint = 0; iEndpoint < Device->Interface.bNumEndpoints; iEndpoint++) {
				UCHAR dt = Body->Endpoints[iEndpoint].bDescriptorType;
				if (dt == USB_DT_ENDPOINT || dt == USB_DT_INTERFACE) break;
			}
			Device->Interface.bNumEndpoints -= iEndpoint;
#if 0
			for (ULONG i = 0; i < Device->Interface.bNumEndpoints; i++) {
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[i], Endpoints[iEndpoint + i], bLength);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[i], Endpoints[iEndpoint + i], bDescriptorType);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[i], Endpoints[iEndpoint + i], bEndpointAddress);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[i], Endpoints[iEndpoint + i], bmAttributes);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[i], Endpoints[iEndpoint + i], wMaxPacketSize);
				NATIVE_COPY_FROM_OFFSET_OFFSET(Device, Body, BodyOffset, Endpoints[i], Endpoints[iEndpoint + i], bInterval);
			}
#endif
			memcpy(Device->Endpoints, &Body->Endpoints[iEndpoint], sizeof(Device->Endpoints[0]) * Device->Interface.bNumEndpoints);
		}
	}

#if 0
	for (ULONG iConf = 0; iConf < Device->Desc.bNumConfigurations; iConf++) {
		PUSB_CONFIGURATION_DESC Config = &Device->Configs[iConf];
		RtlCopyMemory(&Config->Desc, &Body->Config, sizeof(Config->Desc));
		// IOS claims each interface is its own device
		if (Config->Desc.bNumInterfaces == 0) continue;
		Config->Desc.bNumInterfaces = 1;
		PUSB_INTERFACE_DESC Interface = &Config->Interface;
		RtlCopyMemory(&Interface->Desc, &Body->Interface, sizeof(Interface->Desc));
		// Skip vendor and class specific descriptors.
		PUSB_ENDPOINT pEndpoints = &Body->Endpoints[0];
		Interface->ExtraSize = UlpFindNextEndpoint(pEndpoints, (ULONG)Res + ResLength - (ULONG)pEndpoints, 3);
		if (Interface->ExtraSize != 0) {
			Interface->Extra = ExAllocatePool(PagedPool, Interface->ExtraSize);
			if (Interface->Extra == NULL) {
				Status = STATUS_NO_MEMORY;
				break;
			}
			RtlCopyMemory(Interface->Extra, pEndpoints, Interface->ExtraSize);
			pEndpoints = (PUSB_ENDPOINT)((ULONG)pEndpoints + Interface->ExtraSize);
			Interface->Desc.bNumEndpoints -= (Interface->ExtraSize / sizeof(USB_ENDPOINT));
		}

		if (Interface->Desc.bNumEndpoints == 0) continue;
		Interface->Endpoints = ExAllocatePool(PagedPool, Interface->Desc.bNumEndpoints * sizeof(*Interface->Endpoints));
		if (Interface->Endpoints == NULL) {
			Status = STATUS_NO_MEMORY;
			break;
		}

		for (ULONG iEndpoint = 0; iEndpoint < Interface->Desc.bNumEndpoints; iEndpoint++) {
			PUSB_ENDPOINT Endpoint = &Interface->Endpoints[iEndpoint];
			RtlCopyMemory(Endpoint, &Body->Endpoints[iEndpoint], sizeof(*Endpoint));
		}
	}
#endif

	PxiIopFree(Res);
	return Status;
}

LONG UlGetGenericDescriptor(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR Type,
	UCHAR Index,
	UCHAR Interface,
	PVOID Data,
	ULONG Size
) {
	PVOID Buffer = PxiIopAlloc(Size);
	if (Buffer == NULL) return -1;

	LONG Status = UlpGetDesc(DeviceHandle, Buffer, Type, Index, Interface, Size);
	if (Status >= 0) {
		memcpy(Data, Buffer, Size);
	}

	PxiIopFree(Buffer);
	return Status;
}

LONG UlGetHidDescriptor(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR Interface,
	PUSB_HID Hid,
	ULONG Size
) {
	if (Size < sizeof(USB_HID)) return -1;

	return UlGetGenericDescriptor(DeviceHandle, USB_DT_HID, 0, Interface, Hid, Size);
}

LONG UlGetReportDescriptorSize(IOS_USB_HANDLE DeviceHandle, UCHAR Interface, PUSHORT Length) {
	USB_HID Hid;
	LONG Status = UlGetHidDescriptor(DeviceHandle, Interface, &Hid, sizeof(Hid));
	if (Status < 0) return Status;

	if (Hid.bLength > sizeof(Hid)) return -1;

	for (ULONG i = 0; i < Hid.bNumDescriptors; i++) {
		if (Hid.Descriptors[i].bDescriptorType == USB_DT_REPORT) {
			*Length = Hid.Descriptors[i].wDescriptorLength;
			return 0;
		}
	}
	return -1;
}

LONG UlGetReportDescriptor(IOS_USB_HANDLE DeviceHandle, UCHAR Interface, PVOID Data, USHORT Length) {
	if (Data == NULL || Length < USB_DT_MINREPORT_SIZE) return -1;
	return UlGetGenericDescriptor(DeviceHandle, USB_DT_REPORT, 0, Interface, Data, Length);
}

LONG UlGetAsciiString(IOS_USB_HANDLE DeviceHandle, UCHAR Index, USHORT LangID, USHORT Length, PVOID Data, PUSHORT WrittenLength) {
	if (Length > USB_MAX_STRING_LENGTH) Length = USB_MAX_STRING_LENGTH;

	PUCHAR Buf = (PUCHAR)PxiIopAlloc(USB_MAX_STRING_LENGTH);
	if (Buf == NULL) return -1;

	LONG Status = UlpGetDesc(DeviceHandle, Buf, USB_DT_STRING, Index, LangID, USB_MAX_STRING_LENGTH);

	if (Status >= 0) {
		if (Index == 0) {
			// List of supported languages.
			memcpy(Data, Buf, Length);
		}
		else {
			// Convert UTF-16LE to ASCII.
			UCHAR UnicodeIndex = 2;
			UCHAR AsciiIndex = 0;
			PUCHAR Data8 = (PUCHAR)Data;
			UCHAR LastIndex = Buf[0];
			while (AsciiIndex < (Length - 1) && UnicodeIndex < LastIndex) {
				UCHAR Value = Buf[UnicodeIndex];
				if (Buf[UnicodeIndex + 1] != 0) Value = '?';
				Data8[AsciiIndex] = Value;
				AsciiIndex++;
				UnicodeIndex += 2;
			}
			// Null terminate.
			Data8[AsciiIndex] = 0;
			Length = AsciiIndex - 1;
		}
		if (WrittenLength != NULL) *WrittenLength = Length;
	}
	PxiIopFree(Buf);
	return Status;
}

LONG UlTransferIsoMessage(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, UCHAR Packets, PU16BE PacketSizes, PVOID Data) {
	return UlpSendIsoMessage(DeviceHandle, Endpoint, Packets, PacketSizes, Data, false);
}

LONG UlTransferIsoMessageAsync(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, UCHAR Packets, PU16BE PacketSizes, PVOID Data) {
	return UlpSendIsoMessage(DeviceHandle, Endpoint, Packets, PacketSizes, Data, true);
}

LONG UlTransferInterruptMessage(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data) {
	return UlpSendInterruptMessage(DeviceHandle, Endpoint, Length, Data, false);
}

LONG UlTransferInterruptMessageAsync(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data) {
	return UlpSendInterruptMessage(DeviceHandle, Endpoint, Length, Data, true);
}

LONG UlTransferBulkMessage(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data) {
	return UlpSendBulkMessage(DeviceHandle, Endpoint, Length, Data, false);
}

LONG UlTransferBulkMessageAsync(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data) {
	return UlpSendBulkMessage(DeviceHandle, Endpoint, Length, Data, true);
}

LONG UlTransferControlMessage(IOS_USB_HANDLE DeviceHandle, UCHAR RequestType, UCHAR Request, USHORT Value, USHORT Index, USHORT Length, PVOID Data) {
	return UlpSendControlMessage(DeviceHandle, RequestType, Request, Value, Index, Length, Data, false);
}

LONG UlTransferControlMessageAsync(IOS_USB_HANDLE DeviceHandle, UCHAR RequestType, UCHAR Request, USHORT Value, USHORT Index, USHORT Length, PVOID Data) {
	return UlpSendControlMessage(DeviceHandle, RequestType, Request, Value, Index, Length, Data, true);
}

void UlGetDeviceList(PIOS_USB_DEVICE_ENTRY Entry, UCHAR Count, UCHAR InterfaceClass, PUCHAR WrittenCount) {
	UCHAR Wrote = 0;
	IOS_USB_DEVICE_ENTRY LocalEntry;
	if (InterfaceClass != USB_CLASS_HID) {
		ULONG i = 0;
		while (i < USB_COUNT_DEVICES && Wrote < Count && NativeReadBase32(MMIO_OFFSET(s_AttachedDevicesVen, Entries[i].DeviceHandle)) != 0) {
			// Pointer could be in uncached DDR, write to stack then copy
			LocalEntry.DeviceHandle = NativeReadBase32(MMIO_OFFSET(s_AttachedDevicesVen, Entries[i].DeviceHandle));
			LocalEntry.VendorId = NativeReadBase16(MMIO_OFFSET(s_AttachedDevicesVen, Entries[i].VendorId));
			LocalEntry.ProductId = NativeReadBase16(MMIO_OFFSET(s_AttachedDevicesVen, Entries[i].ProductId));
			LocalEntry.Token = NativeReadBase32(MMIO_OFFSET(s_AttachedDevicesVen, Entries[i].Token));
			memcpy32(&Entry[Wrote], &LocalEntry, sizeof(LocalEntry));
			Wrote++;
			i++;
		}
	}
	if (InterfaceClass == 0 || InterfaceClass == USB_CLASS_HID) {
		ULONG i = 0;
		while (i < USB_COUNT_DEVICES && Wrote < Count && NativeReadBase32(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].DeviceHandle)) != 0) {
			// Pointer could be in uncached DDR, write to stack then copy
			LocalEntry.DeviceHandle = NativeReadBase32(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].DeviceHandle));
			LocalEntry.VendorId = NativeReadBase16(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].VendorId));
			LocalEntry.ProductId = NativeReadBase16(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].ProductId));
			LocalEntry.Token = NativeReadBase32(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].Token));
			memcpy32(&Entry[Wrote], &LocalEntry, sizeof(LocalEntry));
			Wrote++;
			i++;
		}
	}
	if (WrittenCount != NULL) *WrittenCount = Wrote;
}

LONG UlSetConfiguration(IOS_USB_HANDLE DeviceHandle, UCHAR Configuration) {
	return UlpSendControlMessage(
		DeviceHandle,
		(USB_CTRLTYPE_DIR_HOST2DEVICE | USB_CTRLTYPE_TYPE_STANDARD | USB_CTRLTYPE_REC_DEVICE),
		USB_REQ_SETCONFIG,
		Configuration,
		0,
		0,
		NULL,
		false
	);
}

LONG UlGetConfiguration(IOS_USB_HANDLE DeviceHandle, PUCHAR Configuration) {
	PUCHAR Buffer = (PUCHAR)PxiIopAlloc(sizeof(UCHAR));
	if (Buffer == NULL) return -1;
	LONG Status = UlpSendControlMessage(
		DeviceHandle,
		(USB_CTRLTYPE_DIR_DEVICE2HOST | USB_CTRLTYPE_TYPE_STANDARD | USB_CTRLTYPE_REC_DEVICE),
		USB_REQ_GETCONFIG,
		0,
		0,
		sizeof(UCHAR),
		Buffer,
		false
	);
	if (Status >= 0) *Configuration = Buffer[0];
	PxiIopFree(Buffer);
	return Status;
}

LONG UlSetAlternativeInterface(IOS_USB_HANDLE DeviceHandle, UCHAR Interface, UCHAR AlternateSetting) {
	return UlpSendControlMessage(
		DeviceHandle,
		(USB_CTRLTYPE_DIR_HOST2DEVICE | USB_CTRLTYPE_TYPE_STANDARD | USB_CTRLTYPE_REC_INTERFACE),
		USB_REQ_SETINTERFACE,
		AlternateSetting,
		Interface,
		0,
		NULL,
		false
	);
}

static LONG UlpCancelEndpoint(IOS_HANDLE Handle, IOS_USB_HANDLE DeviceHandle, PIOS_USB_CANCEL_ENDPOINT_REQ Req, UCHAR Endpoint) {
	STACK_ALIGN(IOS_USB_CANCEL_ENDPOINT_REQ, pReq, 1, 8);
	ZeroMemory32(pReq, sizeof(*pReq));
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, Endpoint, Endpoint);
	memcpy32(Req, pReq, sizeof(*Req));
	return PxiIopIoctl(Handle, USB_IOCTL_CANCEL_ENDPOINT, Req, sizeof(*Req), NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
}

LONG UlCancelEndpoint(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint) {
	PIOS_USB_CANCEL_ENDPOINT_REQ Req = (PIOS_USB_CANCEL_ENDPOINT_REQ)
		PxiIopAlloc(sizeof(IOS_USB_CANCEL_ENDPOINT_REQ));
	if (Req == NULL) return -1;

	IOS_HANDLE Handle = IOS_HANDLE_INVALID;
	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		Handle = s_hUsbVen;
	}
	else if (UlpFindDeviceHid(DeviceHandle) != NULL) {
		Handle = s_hUsbHid;
	}

	LONG Status = -1;
	do {
		if (Handle == IOS_HANDLE_INVALID) break;
		Status = UlpCancelEndpoint(Handle, DeviceHandle, Req, Endpoint);
	} while (0);

	PxiIopFree(Req);
	return Status;
}

LONG UlClearHalt(IOS_USB_HANDLE DeviceHandle) {
	PIOS_USB_CANCEL_ENDPOINT_REQ Req = (PIOS_USB_CANCEL_ENDPOINT_REQ)
		PxiIopAlloc(sizeof(IOS_USB_CANCEL_ENDPOINT_REQ));
	if (Req == NULL) return -1;

	IOS_HANDLE Handle = IOS_HANDLE_INVALID;
	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		Handle = s_hUsbVen;
	}
	else if (UlpFindDeviceHid(DeviceHandle) != NULL) {
		Handle = s_hUsbHid;
	}

	LONG Status = -1;
	do {
		if (Handle == IOS_HANDLE_INVALID) break;
		// Cancel control messages
		Status = UlpCancelEndpoint(Handle, DeviceHandle, Req, USB_CANCEL_CONTROL);
		if (Status < 0) break;
		// Cancel incoming messages
		Status = UlpCancelEndpoint(Handle, DeviceHandle, Req, USB_CANCEL_INCOMING);
		if (Status < 0) break;
		// Cancel outgoing messages
		Status = UlpCancelEndpoint(Handle, DeviceHandle, Req, USB_CANCEL_OUTGOING);
	} while (0);

	PxiIopFree(Req);
	return Status;
}

PVOID UlGetPassedAsyncContext(PVOID AsyncContext) {
	if (AsyncContext == NULL) return NULL;
	PIOS_USB_CONTROL_TRANSFER_REQ Req = (PIOS_USB_CONTROL_TRANSFER_REQ)
		AsyncContext;
	PVOID Ret = Req->Context;

	PxiIopFree(Req);
	return Ret;
}
