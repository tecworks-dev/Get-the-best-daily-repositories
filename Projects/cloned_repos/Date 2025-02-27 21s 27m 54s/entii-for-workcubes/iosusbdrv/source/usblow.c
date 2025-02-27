// USB_VEN low level.
#define DEVL 1
#include <ntddk.h>
#include "usblow.h"
#include "iosusblow.h"
#include "iosapi.h"

#include <stdio.h>

#define ALIGN_TO_ULONG(val) (((ULONG)((val)) + 3) & ~3)

static IOS_HANDLE s_hUsbVen = IOS_HANDLE_INVALID;
static IOS_HANDLE s_hUsbHid = IOS_HANDLE_INVALID;

static const char sc_UsbVen[] ARC_ALIGNED(32) = STRING_BYTESWAP("/dev/usb/ven");
static const char sc_UsbHid[] ARC_ALIGNED(32) = STRING_BYTESWAP("/dev/usb/hid");

static PIOS_USB_DEVICE_CHANGE s_AttachedDevicesVen = NULL;
static PIOS_USB_DEVICE_CHANGE s_AttachedDevicesHid = NULL;
static volatile BOOLEAN s_KnownDevicesVen = FALSE;
static volatile BOOLEAN s_KnownDevicesHid = FALSE;
//static HANDLE s_DeviceChangeThread = NULL;

typedef struct _USB_OPENED_DEVICE {
	IOS_USB_HANDLE DeviceHandle;
	ULONG RefCount;
} USB_OPENED_DEVICE, *PUSB_OPENED_DEVICE;
static USB_OPENED_DEVICE s_OpenedDevices[USB_COUNT_DEVICES] = {0};

static void UlpAsyncInit(PIOS_USB_ASYNC_RESULT Async, PVOID Buffer) {
	KeInitializeEvent(&Async->Event, NotificationEvent, FALSE);
	Async->Status = STATUS_PENDING;
	Async->Buffer = Buffer;
}

typedef enum {
	VARIANT_VEN = 1,
	VARIANT_HID = 2
} USB_INTERNAL_ASYNC_VARIANT;

#if 0
typedef enum {
	THREAD_STATE_DEVICE_CHANGE,
	THREAD_STATE_ATTACH_FINISH
} DEVICE_CHANGE_THREAD_STATE;

static NTSTATUS UlpDeviceChangeIoctl(
	IOS_HANDLE Handle,
	DEVICE_CHANGE_THREAD_STATE State,
	PIOS_USB_DEVICE_CHANGE Attached,
	PIOS_USB_ASYNC_RESULT Async
) {
	if (State == THREAD_STATE_DEVICE_CHANGE)
		return HalIopIoctlAsync(
			Handle,
			USB_IOCTL_DEVICE_CHANGE,
			NULL, 0,
			Attached, sizeof(*Attached),
			&Async->Status, &Async->Event
		);
	else if (State == THREAD_STATE_ATTACH_FINISH)
		return HalIopIoctlAsync(
			Handle,
			USB_IOCTL_ATTACH_FINISH,
			NULL, 0, NULL, 0,
			&Async->Status, &Async->Event
		);
	return STATUS_INVALID_PARAMETER;
}

static DEVICE_CHANGE_THREAD_STATE UlpDeviceChangeState(DEVICE_CHANGE_THREAD_STATE State) {
	if (State == THREAD_STATE_DEVICE_CHANGE) return THREAD_STATE_ATTACH_FINISH;
	return THREAD_STATE_DEVICE_CHANGE;
}
#endif

static void UlpDeviceChangeCallback(NTSTATUS Status, ULONG Result, PVOID Context);

static void UlpAttachFinishCallback(NTSTATUS Status, ULONG Result, PVOID Context) {
	USB_INTERNAL_ASYNC_VARIANT Variant = (USB_INTERNAL_ASYNC_VARIANT)Context;
	
	if (!NT_SUCCESS(Status)) return;
	
	PIOS_USB_DEVICE_CHANGE Devices = NULL;
	IOS_HANDLE Handle;
	if (Variant == VARIANT_VEN) {
		Devices = s_AttachedDevicesVen;
		Handle = s_hUsbVen;
	} else if (Variant == VARIANT_HID) {
		Devices = s_AttachedDevicesHid;
		Handle = s_hUsbHid;
	}
	else return; // wtf?
	
	// Call DeviceChange.
	HalIopIoctlAsyncDpc(
		Handle,
		USB_IOCTL_DEVICE_CHANGE,
		NULL, 0,
		Devices, sizeof(*Devices),
		IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
		UlpAttachFinishCallback,
		Context
	);
}

static void UlpDeviceChangeCallback(NTSTATUS Status, ULONG Result, PVOID Context) {
	USB_INTERNAL_ASYNC_VARIANT Variant = (USB_INTERNAL_ASYNC_VARIANT)Context;
	
	if (!NT_SUCCESS(Status)) return;
	
	PIOS_USB_DEVICE_CHANGE Devices = NULL;
	IOS_HANDLE Handle;
	//CHAR Buffer[512];
	BOOLEAN KnownDevices;
	if (Variant == VARIANT_VEN) {
		Devices = s_AttachedDevicesVen;
		Handle = s_hUsbVen;
		KnownDevices = s_KnownDevicesVen;
#if 0 // do we actually need this?
		if (!s_KnownDevicesVen) {
			//_snprintf(Buffer, sizeof(Buffer), "USBVEN: Number of attached devices: %d\n", Result);
			//HalDisplayString(Buffer);
			KeStallExecutionProcessor(100);
		}
#endif
	} else if (Variant == VARIANT_HID) {
		Devices = s_AttachedDevicesHid;
		Handle = s_hUsbHid;
		KnownDevices = s_KnownDevicesHid;
#if 0 // do we actually need this?
		if (!s_KnownDevicesHid) {
			//_snprintf(Buffer, sizeof(Buffer), "USBVEN: Number of attached devices: %d\n", Result);
			//HalDisplayString(Buffer);
			KeStallExecutionProcessor(100);
		}
#endif
	}
	else return; // wtf?
	
	// when ARC doesn't restart IOS, first DeviceChange call will always give a response of zero devices, even though old device handles are still valid
	//if (Result != 0 || KnownDevices)
	{
		// Zero out unused entries.
		// Do this using NativeWriteBase32 because endianness reasons.
		ULONG Length = sizeof(Devices->Entries[0]) * (USB_COUNT_DEVICES - Result);
		Length /= sizeof(ULONG);
		for (ULONG i = 0; i < Length; i++) {
			NativeWriteBase32(Devices, __builtin_offsetof(IOS_USB_DEVICE_CHANGE, Entries[Result]) + (i * sizeof(ULONG)), 0);
		}
	}
	
	if (Variant == VARIANT_VEN) s_KnownDevicesVen = TRUE;
	if (Variant == VARIANT_HID) s_KnownDevicesHid = TRUE;
	
	// Call AttachFinish.
	HalIopIoctlAsyncDpc(
		Handle,
		USB_IOCTL_ATTACH_FINISH,
		NULL, 0, NULL, 0,
		IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
		UlpAttachFinishCallback,
		Context
	);
}

#if 0
static void UlpDeviceChangeThread(PVOID Context) {
	(void)Context;
	
	IOS_USB_ASYNC_RESULT AsyncVen, AsyncHid;
	UlpAsyncInit(&AsyncVen, NULL);
	UlpAsyncInit(&AsyncHid, NULL);
	BOOLEAN Continue = FALSE;
	NTSTATUS StatusVen = STATUS_SUCCESS;
	NTSTATUS StatusHid = STATUS_SUCCESS;
	
	PVOID Objects[] = { &AsyncVen.Event, &AsyncHid.Event };
	BOOLEAN IoctlVen = TRUE, IoctlHid = TRUE;
	DEVICE_CHANGE_THREAD_STATE
		StateVen = THREAD_STATE_DEVICE_CHANGE,
		StateHid = THREAD_STATE_DEVICE_CHANGE;
	
	do {
		Continue = FALSE;
		if (
			NT_SUCCESS(StatusVen) &&
			s_AttachedDevicesVen == NULL &&
			s_hUsbVen != IOS_HANDLE_INVALID &&
			IoctlVen
		) {
			Continue = TRUE;
			IoctlVen = FALSE;
			StatusVen = UlpDeviceChangeIoctl(
				s_hUsbVen,
				StateVen,
				s_AttachedDevicesVen,
				&AsyncVen
			);
		}
		if (
			NT_SUCCESS(StatusHid) &&
			s_AttachedDevicesHid &&
			s_hUsbHid != IOS_HANDLE_INVALID &&
			IoctlHid
		) {
			Continue = TRUE;
			IoctlHid = FALSE;
			StatusVen = UlpDeviceChangeIoctl(
				s_hUsbHid,
				StateHid,
				s_AttachedDevicesHid,
				&AsyncHid
			);
		}
		
		// If no ioctls were sent, then exit the thread.
		if (!Continue) break;
		
		// If either IoctlAsync call failed, that event won't ever be set.
		// Therefore it is safe to do this.
		
		NTSTATUS Status = KeWaitForMultipleObjects(
			2, Objects, WaitAny, Executive, KernelMode, FALSE,
			NULL, NULL
		);
		
		if (!NT_SUCCESS(Status)) break;
		if (Status == STATUS_WAIT_0) {
			IoctlVen = TRUE;
			StatusVen = AsyncVen.Status;
			if (StateVen == THREAD_STATE_DEVICE_CHANGE && NT_SUCCESS(StatusVen)) {
				KeSetEvent(&s_DevicesKnownVen, 0, FALSE);
			}
			StateVen = UlpDeviceChangeState(StateVen);
			UlpAsyncInit(&AsyncVen, NULL);
			continue;
		}
		if (Status == STATUS_WAIT_1) {
			IoctlHid = TRUE;
			StatusHid = AsyncHid.Status;
			if (StateHid == THREAD_STATE_DEVICE_CHANGE && NT_SUCCESS(StatusHid)) {
				KeSetEvent(&s_DevicesKnownHid, 0, FALSE);
			}
			StateHid = UlpDeviceChangeState(StateHid);
			UlpAsyncInit(&AsyncHid, NULL);
			continue;
		}
		// ???
		break;
	} while (TRUE);
	
	PsTerminateSystemThread(STATUS_SUCCESS);
}
#endif

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

static NTSTATUS UlpIoctl(IOS_USB_HANDLE DeviceHandle, ULONG ControlCode, PVOID Input, ULONG LengthInput, PVOID Output, ULONG LengthOutput, PIOS_USB_ASYNC_RESULT Async, IOP_CALLBACK Callback, PVOID Context) {
	if (Async != NULL && Callback != NULL) return STATUS_INVALID_PARAMETER;
	IOS_HANDLE Handle = UlpGetIosForUsb(DeviceHandle);
	if (Async != NULL) return HalIopIoctlAsync(Handle, ControlCode, Input, LengthInput, Output, LengthOutput, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE, &Async->Status, &Async->Event);
	else if (Callback != NULL) return HalIopIoctlAsyncDpc(Handle, ControlCode, Input, LengthInput, Output, LengthOutput, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE, Callback, Context);
	else return HalIopIoctl(Handle, ControlCode, Input, LengthInput, Output, LengthOutput, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
}

static NTSTATUS UlpIoctlv(IOS_USB_HANDLE DeviceHandle, ULONG ControlCode, ULONG NumRead, ULONG NumWritten, PIOS_IOCTL_VECTOR Buffers, BOOLEAN EndpointIn, PIOS_USB_ASYNC_RESULT Async, IOP_CALLBACK Callback, PVOID Context) {
	if (Async != NULL && Callback != NULL) return STATUS_INVALID_PARAMETER;
	IOS_HANDLE Handle = UlpGetIosForUsb(DeviceHandle);
	// for an endpoint out (write), swap in and out
	// for an endpoint in (read), swap out 
	ULONG SwapIn = 0;
	if (!EndpointIn) SwapIn = BIT(1);
	ULONG SwapOut = BIT(1);
	if (Buffers[2].Pointer != NULL) {
		// iso message, buffer 1 is array of u16 packet sizes in wrong endianness, swapping will place everything in correct place and correct endianness for IOP
		SwapIn |= BIT(1);
		if (!EndpointIn) SwapIn |= BIT(2);
		SwapOut |= BIT(2);
	}
	
	if (Async != NULL) return HalIopIoctlvAsync(Handle, ControlCode, NumRead, NumWritten, Buffers, SwapIn, SwapOut, &Async->Status, &Async->Event);
	else if (Callback != NULL) return HalIopIoctlvAsyncDpc(Handle, ControlCode, NumRead, NumWritten, Buffers, SwapIn, SwapOut, Callback, Context);
	else return HalIopIoctlv(Handle, ControlCode, NumRead, NumWritten, Buffers, SwapIn, SwapOut);
}

static NTSTATUS UlpSendIsoMessage(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR bEndpoint,
	UCHAR bPackets,
	PU16BE rpPacketSizes,
	PVOID rpData,
	PIOS_USB_ASYNC_RESULT Async,
	IOP_CALLBACK Callback,
	PVOID Context
) {
	if (rpPacketSizes == NULL) return STATUS_INVALID_PARAMETER;
	if (rpData == NULL) return STATUS_INVALID_PARAMETER;
	
	USHORT wLength = 0;
	for (ULONG i = 0; i < bPackets; i++) wLength += rpPacketSizes[i].v;
	if (wLength == 0) return STATUS_INVALID_PARAMETER;
	
	PIOS_USB_ISOCHRONOUS_TRANSFER_REQ Req = 
		(PIOS_USB_ISOCHRONOUS_TRANSFER_REQ)
		HalIopAlloc(sizeof(IOS_USB_ISOCHRONOUS_TRANSFER_REQ));
	if (Req == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	// Work around the requirement of 32-bit writes to uncached DDR mappings.
	// RtlCopyMemory is guaranteed to always use 32-bit writes;
	// as sizeof(Req) is 32 bit aligned.
	IOS_USB_ISOCHRONOUS_TRANSFER_REQ _Req;
	PIOS_USB_ISOCHRONOUS_TRANSFER_REQ pReq = &_Req;
	RtlZeroMemory(pReq, sizeof(Req));
	
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, NumberOfPackets, bPackets);
	NATIVE_WRITE(pReq, EndpointAddress, bEndpoint);
	
	pReq->Vectors[0].Pointer = Req;
	pReq->Vectors[0].Length = sizeof(*Req);
	pReq->Vectors[1].Pointer = rpPacketSizes;
	pReq->Vectors[1].Length = sizeof(USHORT) * bPackets;
	pReq->Vectors[2].Pointer = rpData;
	pReq->Vectors[2].Length = wLength;
	
	pReq->Context = Context;
	
	RtlCopyMemory(Req, pReq, sizeof(*Req));
	
	BOOLEAN EndpointIn = (bEndpoint & USB_ENDPOINT_IN) != 0;
	
	if (Async != NULL) UlpAsyncInit(Async, Req);
	NTSTATUS Status = UlpIoctlv(DeviceHandle, USB_IOCTLV_ISOCHRONOUS_TRANSFER, 1, 2, Req->Vectors, EndpointIn, Async, Callback, Req);
	if ((Async == NULL && Callback == NULL) || !NT_SUCCESS(Status)) HalIopFree(Req);
	
	return Status;
}

static NTSTATUS UlpSendControlMessage(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR bmRequestType,
	UCHAR bRequest,
	USHORT wValue,
	USHORT wIndex,
	USHORT wLength,
	PVOID rpData,
	PIOS_USB_ASYNC_RESULT Async,
	IOP_CALLBACK Callback,
	PVOID Context
) {
	if (rpData == NULL && wLength != 0) return STATUS_INVALID_PARAMETER;
	if (wLength == 0 && rpData != NULL) return STATUS_INVALID_PARAMETER;
	
	PIOS_USB_CONTROL_TRANSFER_REQ Req = (PIOS_USB_CONTROL_TRANSFER_REQ)
		HalIopAlloc(sizeof(IOS_USB_CONTROL_TRANSFER_REQ));
	if (Req == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	// Work around the requirement of 32-bit writes to uncached DDR mappings.
	// RtlCopyMemory is guaranteed to always use 32-bit writes;
	// as sizeof(Req) is 32 bit aligned.
	IOS_USB_CONTROL_TRANSFER_REQ _Req;
	PIOS_USB_CONTROL_TRANSFER_REQ pReq = &_Req;

	RtlZeroMemory(pReq, sizeof(*Req));
	
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, bmRequestType, bmRequestType);
	NATIVE_WRITE(pReq, bRequest, bRequest);
	NATIVE_WRITE(pReq, wValue, wValue);
	NATIVE_WRITE(pReq, wIndex, wIndex);
	
	pReq->Vectors[0].Pointer = Req;
	pReq->Vectors[0].Length = sizeof(*Req);
	pReq->Vectors[1].Pointer = rpData;
	pReq->Vectors[1].Length = wLength;
	
	pReq->Context = Context;
	
	RtlCopyMemory(Req, pReq, sizeof(*Req));
	
#if 0
	{
		CHAR DebugText[512];
		PUCHAR Req8 = (PUCHAR)(ULONG)Req;
		for (int i = 0; i < sizeof(*Req); i += 0x10) {
			_snprintf(DebugText, sizeof(DebugText),
				"%02x %02x %02x %02x %02x %02x %02x %02x "
				"%02x %02x %02x %02x %02x %02x %02x %02x\n",
				Req8[i + 0x0], Req8[i + 0x1],
				Req8[i + 0x2], Req8[i + 0x3],
				Req8[i + 0x4], Req8[i + 0x5],
				Req8[i + 0x6], Req8[i + 0x7],
				Req8[i + 0x8], Req8[i + 0x9],
				Req8[i + 0xa], Req8[i + 0xb],
				Req8[i + 0xc], Req8[i + 0xd],
				Req8[i + 0xe], Req8[i + 0xf]
			);
			HalDisplayString(DebugText);
		}
		HalDisplayString("\n");
	}
#endif

	BOOLEAN EndpointIn = (bmRequestType & USB_CTRLTYPE_DIR_DEVICE2HOST) == 0;
	
	if (Async != NULL) UlpAsyncInit(Async, Req);
	NTSTATUS Status = UlpIoctlv(DeviceHandle, USB_IOCTLV_CONTROL_TRANSFER, 1, 1, Req->Vectors, EndpointIn, Async, Callback, Req);
	if ((Async == NULL && Callback == NULL) || !NT_SUCCESS(Status)) HalIopFree(Req);
	
	return Status;
}

static NTSTATUS UlpSendBulkMessage(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR bEndpoint,
	USHORT wLength,
	PVOID rpData,
	PIOS_USB_ASYNC_RESULT Async,
	IOP_CALLBACK Callback,
	PVOID Context
) {
	
	if (rpData == NULL && wLength != 0) return STATUS_INVALID_PARAMETER;
	if (wLength == 0 && rpData != NULL) return STATUS_INVALID_PARAMETER;
	
	PIOS_USB_BULK_TRANSFER_REQ Req = (PIOS_USB_BULK_TRANSFER_REQ)
		HalIopAlloc(sizeof(IOS_USB_BULK_TRANSFER_REQ));
	if (Req == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	// Work around the requirement of 32-bit writes to uncached DDR mappings.
	// RtlCopyMemory is guaranteed to always use 32-bit writes;
	// as sizeof(Req) is 32 bit aligned.
	IOS_USB_BULK_TRANSFER_REQ _Req;
	PIOS_USB_BULK_TRANSFER_REQ pReq = &_Req;

	RtlZeroMemory(pReq, sizeof(*Req));
	
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, EndpointAddress, bEndpoint);
	
	pReq->Vectors[0].Pointer = Req;
	pReq->Vectors[0].Length = sizeof(*Req);
	pReq->Vectors[1].Pointer = rpData;
	pReq->Vectors[1].Length = wLength;
		
	pReq->Context = Context;
	RtlCopyMemory(Req, pReq, sizeof(*Req));
	
	BOOLEAN EndpointIn = (bEndpoint & USB_ENDPOINT_IN) != 0;
	
	if (Async != NULL) UlpAsyncInit(Async, Req);
	NTSTATUS Status = UlpIoctlv(DeviceHandle, USB_IOCTLV_BULK_TRANSFER, 1, 1, Req->Vectors, EndpointIn, Async, Callback, Req);
	if ((Async == NULL && Callback == NULL) || !NT_SUCCESS(Status)) HalIopFree(Req);
	
	return Status;
}

static NTSTATUS UlpSendInterruptMessage(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR bEndpoint,
	USHORT wLength,
	PVOID rpData,
	PIOS_USB_ASYNC_RESULT Async,
	IOP_CALLBACK Callback,
	PVOID Context
) {
	
	if (rpData == NULL && wLength != 0) return STATUS_INVALID_PARAMETER;
	if (wLength == 0 && rpData != NULL) return STATUS_INVALID_PARAMETER;
	
	PIOS_USB_INTERRUPT_TRANSFER_REQ Req = (PIOS_USB_INTERRUPT_TRANSFER_REQ)
		HalIopAlloc(sizeof(IOS_USB_INTERRUPT_TRANSFER_REQ));
	if (Req == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	// Work around the requirement of 32-bit writes to uncached DDR mappings.
	// RtlCopyMemory is guaranteed to always use 32-bit writes;
	// as sizeof(Req) is 32 bit aligned.
	IOS_USB_INTERRUPT_TRANSFER_REQ _Req;
	PIOS_USB_INTERRUPT_TRANSFER_REQ pReq = &_Req;

	RtlZeroMemory(pReq, sizeof(*Req));
	
	
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, Write, ((bEndpoint & USB_ENDPOINT_IN) == 0));
	NATIVE_WRITE(pReq, EndpointAddress, bEndpoint);
	pReq->Vectors[0].Pointer = Req;
	pReq->Vectors[0].Length = sizeof(*Req);
	pReq->Vectors[1].Pointer = rpData;
	pReq->Vectors[1].Length = wLength;
	
	pReq->Context = Context;
	RtlCopyMemory(Req, pReq, sizeof(*Req));
	
	BOOLEAN EndpointIn = (bEndpoint & USB_ENDPOINT_IN) != 0;
	
	if (Async != NULL) UlpAsyncInit(Async, Req);
	//HalDisplayString("Ioctlv InterruptTransfer\n");
	NTSTATUS Status = UlpIoctlv(DeviceHandle, USB_IOCTLV_INTERRUPT_TRANSFER, 1, 1, Req->Vectors, EndpointIn, Async, Callback, Req);
	if ((Async == NULL && Callback == NULL) || !NT_SUCCESS(Status)) HalIopFree(Req);
	
	return Status;
}

static inline NTSTATUS UlpGetDesc(
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
		NULL,
		NULL,
		NULL
	);
}

#if 0
static ULONG UlpFindNextEndpoint(PVOID Buffer, LONG Size, UCHAR Align) {
	PUCHAR Ptr = (PUCHAR)Buffer;
	
	while (Size > 2 && Ptr[0] != 0) {
		if (Ptr[1] == USB_DT_ENDPOINT || Ptr[1] == USB_DT_INTERFACE) break;
		
		UCHAR ChunkSize = Ptr[0];
		UCHAR AlignedChunkSize = (ChunkSize + Align) & ~Align;
		Size -= AlignedChunkSize;
		Ptr += AlignedChunkSize;
	}
	
	return ((ULONG)Ptr - (ULONG)Buffer);
}
#endif

static NTSTATUS UlpCheckVersionImpl(IOS_HANDLE Handle, PIOS_USB_VERSION UsbVersion) {
	NTSTATUS Status = HalIopIoctl(Handle, USB_IOCTL_GET_VERSION, NULL, 0, UsbVersion, sizeof(*UsbVersion), IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	if (!NT_SUCCESS(Status)) return STATUS_OBJECT_TYPE_MISMATCH;
	if (NativeReadBase8(MMIO_OFFSET(UsbVersion, Major)) != 5) return STATUS_OBJECT_TYPE_MISMATCH;
	if (NativeReadBase8(MMIO_OFFSET(UsbVersion, Minor)) != 0) return STATUS_OBJECT_TYPE_MISMATCH;
	if (NativeReadBase8(MMIO_OFFSET(UsbVersion, Revision)) != 1) return STATUS_OBJECT_TYPE_MISMATCH;
	return STATUS_SUCCESS;
}

static NTSTATUS UlpCheckVersion(IOS_HANDLE Handle) {
	PIOS_USB_VERSION UsbVersion = HalIopAlloc(sizeof(IOS_USB_VERSION));
	if (UsbVersion == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	NTSTATUS Status = UlpCheckVersionImpl(Handle, UsbVersion);
	HalIopFree(UsbVersion);
	return Status;
}

static NTSTATUS UlpSuspendResume(IOS_USB_HANDLE DeviceHandle, BOOLEAN Resumed) {
	PIOS_USB_SUSPEND_RESUME_REQ Buf = (PIOS_USB_SUSPEND_RESUME_REQ)
		HalIopAlloc(sizeof(IOS_USB_SUSPEND_RESUME_REQ));
	if (Buf == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	IOS_USB_SUSPEND_RESUME_REQ Req;
	PIOS_USB_SUSPEND_RESUME_REQ pReq = &Req;
	RtlZeroMemory(&Req, sizeof(Req));
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, Resume, Resumed);
	RtlCopyMemory(Buf, &Req, sizeof(Req));
	NTSTATUS Status = STATUS_NO_SUCH_DEVICE;
	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		Status = HalIopIoctl(s_hUsbVen, USB_IOCTL_SUSPEND_RESUME, Buf, sizeof(*Buf), NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	} else if (UlpFindDeviceHid(DeviceHandle) != NULL) {
		Status = HalIopIoctl(s_hUsbHid, USB_IOCTL_SUSPEND_RESUME, Buf, sizeof(*Buf), NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	}
	HalIopFree(Buf);
	return Status;
}

static NTSTATUS UlpSuspendDevice(IOS_USB_HANDLE DeviceHandle) {
	return UlpSuspendResume(DeviceHandle, FALSE);
}

static NTSTATUS UlpResumeDevice(IOS_USB_HANDLE DeviceHandle) {
	return UlpSuspendResume(DeviceHandle, TRUE);
}

NTSTATUS UlInit(void) {
	//BOOLEAN InEmulator = (BOOLEAN) (ULONG) RUNTIME_BLOCK[RUNTIME_IN_EMULATOR];
	NTSTATUS Status = STATUS_SUCCESS;
	if (s_AttachedDevicesVen == NULL) {
		if (s_hUsbVen == IOS_HANDLE_INVALID) {
			// Open handle.
			NTSTATUS VenStatus = HalIopOpen(sc_UsbVen, IOSOPEN_NONE, &s_hUsbVen);
			if (!NT_SUCCESS(VenStatus)) {
				return VenStatus;
			}
			// Check version.
			VenStatus = UlpCheckVersion(s_hUsbVen);
			if (!NT_SUCCESS(VenStatus)) {
				// Incorrect version.
				HalIopClose(s_hUsbVen);
				s_hUsbVen = IOS_HANDLE_INVALID;
				return VenStatus;
			}
		}
		// Allocate memory for attached devices.
		s_AttachedDevicesVen = HalIopAlloc(sizeof(*s_AttachedDevicesVen));
		if (s_AttachedDevicesVen == NULL) {
			HalIopClose(s_hUsbVen);
			s_hUsbVen = IOS_HANDLE_INVALID;
			return STATUS_INSUFFICIENT_RESOURCES;
		}
		RtlZeroMemory(s_AttachedDevicesVen, sizeof(*s_AttachedDevicesVen));
	}

	// Unlike libogc implementation, only get here if usbven is open and is v5.
	if (s_AttachedDevicesHid == NULL) {
		NTSTATUS HidStatus;
		if (s_hUsbHid == IOS_HANDLE_INVALID) {
			// Open handle.
			HidStatus = HalIopOpen(sc_UsbHid, IOSOPEN_NONE, &s_hUsbHid);
			if (NT_SUCCESS(HidStatus)) {
				// Check version.
				HidStatus = UlpCheckVersion(s_hUsbHid);
				if (!NT_SUCCESS(HidStatus)) {
					// Incorrect version.
					HalIopClose(s_hUsbHid);
					s_hUsbHid = IOS_HANDLE_INVALID;
				}
			}
		}

		if (s_hUsbHid != IOS_HANDLE_INVALID) {
			// Allocate memory for attached devices.
			s_AttachedDevicesHid = HalIopAlloc(sizeof(*s_AttachedDevicesHid));
			if (s_AttachedDevicesHid == NULL) {
				HalIopClose(s_hUsbHid);
				HalIopClose(s_hUsbVen);
				HalIopFree(s_AttachedDevicesVen);
				s_AttachedDevicesVen = NULL;
				s_hUsbHid = IOS_HANDLE_INVALID;
				s_hUsbVen = IOS_HANDLE_INVALID;
				return STATUS_INSUFFICIENT_RESOURCES;
			}
			RtlZeroMemory(s_AttachedDevicesHid, sizeof(*s_AttachedDevicesHid));
		} else {
			HalIopClose(s_hUsbVen);
			HalIopFree(s_AttachedDevicesVen);
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
	if (NT_SUCCESS(Status)) {
		RtlZeroMemory(s_OpenedDevices, sizeof(s_OpenedDevices));
		//PIOS_USB_DEVICE_CHANGE DeviceChange = (PIOS_USB_DEVICE_CHANGE) RUNTIME_BLOCK[RUNTIME_USB_DEVICES];
		
		//RtlCopyMemory(s_AttachedDevicesVen, &DeviceChange[0], sizeof(DeviceChange[0]));
		//RtlCopyMemory(s_AttachedDevicesHid, &DeviceChange[1], sizeof(DeviceChange[0]));
		//s_KnownDevicesVen = TRUE;
		//s_KnownDevicesHid = TRUE;
		Status = HalIopIoctlAsyncDpc(
			s_hUsbVen,
			USB_IOCTL_DEVICE_CHANGE,
			NULL, 0,
			s_AttachedDevicesVen, sizeof(*s_AttachedDevicesVen),
			IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
			UlpDeviceChangeCallback,
			(PVOID)VARIANT_VEN
		);
	}
	if (NT_SUCCESS(Status)) {
		Status = HalIopIoctlAsyncDpc(
			s_hUsbHid,
			USB_IOCTL_DEVICE_CHANGE,
			NULL, 0,
			s_AttachedDevicesHid, sizeof(*s_AttachedDevicesHid),
			IOCTL_SWAP_NONE, IOCTL_SWAP_NONE,
			UlpDeviceChangeCallback,
			(PVOID)VARIANT_HID
		);
	}
	if (!NT_SUCCESS(Status)) {
		HalIopIoctl(s_hUsbHid, USB_IOCTL_SHUTDOWN, NULL, 0, NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
		HalIopClose(s_hUsbHid);
		HalIopIoctl(s_hUsbVen, USB_IOCTL_SHUTDOWN, NULL, 0, NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
		HalIopClose(s_hUsbVen);
		HalIopFree(s_AttachedDevicesHid);
		s_AttachedDevicesHid = NULL;
		HalIopFree(s_AttachedDevicesVen);
		s_AttachedDevicesVen = NULL;
		s_hUsbHid = IOS_HANDLE_INVALID;
		s_hUsbVen = IOS_HANDLE_INVALID;
	} else {
		// Spin until first device change comes in.
		while (!s_KnownDevicesHid || !s_KnownDevicesVen) { }
	}
	
	return Status;
}

static NTSTATUS UlpGetDescriptorsHidForOpen(IOS_USB_HANDLE DeviceHandle) {
	PIOS_USB_GET_DEVICE_INFO_REQ Req = (PIOS_USB_GET_DEVICE_INFO_REQ)
		HalIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_REQ));
	if (Req == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	IOS_USB_GET_DEVICE_INFO_REQ _Req;
	PIOS_USB_GET_DEVICE_INFO_REQ pReq = &_Req;
	RtlZeroMemory(&_Req, sizeof(_Req));
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, AlternateSetting, 0);
	RtlCopyMemory(Req, &_Req, sizeof(_Req));
	
	PIOS_USB_GET_DEVICE_INFO_HID_RES HidRes = (PIOS_USB_GET_DEVICE_INFO_HID_RES)
		HalIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES));
	if (HidRes == NULL) {
		HalIopFree(Req);
		return STATUS_INSUFFICIENT_RESOURCES;
	}
	
	RtlZeroMemory(HidRes, sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES));
	NTSTATUS Status = HalIopIoctl(s_hUsbHid, USB_IOCTL_GET_DEVICE_INFO, Req, sizeof(*Req), HidRes, sizeof(*HidRes), IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	HalIopFree(Req);
	HalIopFree(HidRes);
	return Status;
}

#if 0
static NTSTATUS UlpResetDevice(IOS_USB_HANDLE DeviceHandle) {
	PIOS_USB_RESET_REQ Buf = (PIOS_USB_RESET_REQ)
		HalIopAlloc(sizeof(IOS_USB_RESET_REQ));
	if (Buf == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	Buf->DeviceHandle = DeviceHandle;
	NTSTATUS Status = STATUS_NO_SUCH_DEVICE;
	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		Status = HalIopIoctl(s_hUsbVen, USB_IOCTL_RESET, Buf, sizeof(*Buf), NULL, 0);
	}
	// HID does not have USB_IOCTL_RESET
	HalIopFree(Buf);
	return Status;
}
#endif

static NTSTATUS UlpClearHaltMsg(IOS_USB_HANDLE DeviceHandle) {
	// some devices can't handle this, so ignore result.
	UlpSendControlMessage(
		DeviceHandle,
		(USB_CTRLTYPE_DIR_HOST2DEVICE | USB_CTRLTYPE_TYPE_STANDARD | USB_CTRLTYPE_REC_ENDPOINT),
		USB_REQ_CLEARFEATURE,
		USB_FEATURE_ENDPOINT_HALT,
		0,
		0,
		NULL,
		NULL,
		NULL,
		NULL
	);
	return STATUS_SUCCESS;
}

static NTSTATUS UlpOpenDevice(IOS_USB_HANDLE DeviceHandle) {
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
		return STATUS_NO_SUCH_DEVICE;
	}
	NTSTATUS Status = UlpResumeDevice(DeviceHandle);
	if (!NT_SUCCESS(Status)) {
		if (Status == STATUS_INVALID_PARAMETER) {
			// Might be opened already (perhaps from ARC firmware). Try suspend then resume.
			Status = UlpSuspendDevice(DeviceHandle);
			if (!NT_SUCCESS(Status)) return Status;
			Status = UlpResumeDevice(DeviceHandle);
			if (!NT_SUCCESS(Status)) return Status;
		}
		return Status;
	}
	//USB_DEVICE_DESC Descriptors;
	Status = UlpGetDescriptorsHidForOpen(DeviceHandle);
	if (!NT_SUCCESS(Status)) {
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

static NTSTATUS UlpOpenDeviceEnsureClearHalt(IOS_USB_HANDLE DeviceHandle) {
	return UlpOpenDevice(DeviceHandle);
}

NTSTATUS UlOpenDevice(IOS_USB_HANDLE DeviceHandle) {
	PUSB_OPENED_DEVICE EmptyDevice = NULL;
	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		IOS_USB_HANDLE Current = s_OpenedDevices[i].DeviceHandle;
		if (Current == DeviceHandle) {
			s_OpenedDevices[i].RefCount++;
			return STATUS_SUCCESS;
		}
		if (s_OpenedDevices[i].RefCount == 0 && Current == 0 && EmptyDevice == NULL) {
			EmptyDevice = &s_OpenedDevices[i];
		}
	}
	if (EmptyDevice == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	NTSTATUS Status = UlpOpenDeviceEnsureClearHalt(DeviceHandle);
	if (!NT_SUCCESS(Status)) return Status;
	EmptyDevice->DeviceHandle = DeviceHandle;
	EmptyDevice->RefCount = 1;
	return Status;
}

NTSTATUS UlCloseDevice(IOS_USB_HANDLE DeviceHandle) {
	for (ULONG i = 0; i < USB_COUNT_DEVICES; i++) {
		IOS_USB_HANDLE Current = s_OpenedDevices[i].DeviceHandle;
		if (Current == DeviceHandle) {
			s_OpenedDevices[i].RefCount--;
			if (s_OpenedDevices[i].RefCount != 0)
				return STATUS_SUCCESS;
			s_OpenedDevices[i].DeviceHandle = 0;
			return UlpSuspendResume(DeviceHandle, FALSE);
		}
	}
	return STATUS_NO_SUCH_DEVICE;
}

NTSTATUS UlGetDeviceDesc(IOS_USB_HANDLE DeviceHandle, PUSB_DEVICE Device) {
	PUSB_DEVICE Buf = HalIopAlloc(USB_DT_DEVICE_SIZE);
	if (Buf == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	NTSTATUS Status = UlpSendControlMessage(DeviceHandle, USB_CTRLTYPE_DIR_DEVICE2HOST, USB_REQ_GETDESCRIPTOR, (USB_DT_DEVICE << 8), 0, USB_DT_DEVICE_SIZE, Buf, NULL, NULL, NULL);
	if (NT_SUCCESS(Status)) RtlCopyMemory(Device, Buf, USB_DT_DEVICE_SIZE);
	HalIopFree(Buf);
	return Status;
}

NTSTATUS UlGetDescriptors(IOS_USB_HANDLE DeviceHandle, PUSB_DEVICE_DESC Device) {
	PIOS_USB_DEVICE_ENTRY Ven = UlpFindDeviceVen(DeviceHandle);
	PIOS_USB_DEVICE_ENTRY Hid = UlpFindDeviceHid(DeviceHandle);
	if (Ven == NULL && Hid == NULL) return STATUS_INVALID_PARAMETER;
	
	PIOS_USB_GET_DEVICE_INFO_RES_BODY Body = NULL;
	
	PIOS_USB_GET_DEVICE_INFO_REQ Req = (PIOS_USB_GET_DEVICE_INFO_REQ)
		HalIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_REQ));
	if (Req == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	{
		IOS_USB_GET_DEVICE_INFO_REQ _Req;
		PIOS_USB_GET_DEVICE_INFO_REQ pReq = &_Req;
		RtlZeroMemory(&_Req, sizeof(_Req));
		NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
		NATIVE_WRITE(pReq, AlternateSetting, 0);
		RtlCopyMemory(Req, &_Req, sizeof(_Req));
	}
	PVOID Res = NULL;
	ULONG ResLength = 0;
	IOS_HANDLE Handle = IOS_HANDLE_INVALID;
	BOOLEAN IsHid = FALSE;
	
	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		PIOS_USB_GET_DEVICE_INFO_VEN_RES VenRes = (PIOS_USB_GET_DEVICE_INFO_VEN_RES)
			HalIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_VEN_RES));
		if (VenRes == NULL) {
			HalIopFree(Req);
			return STATUS_INSUFFICIENT_RESOURCES;
		}
		Res = VenRes;
		ResLength = sizeof(IOS_USB_GET_DEVICE_INFO_VEN_RES);
		Body = (PIOS_USB_GET_DEVICE_INFO_RES_BODY)&VenRes->Device;
		Handle = s_hUsbVen;
	} else if (UlpFindDeviceHid(DeviceHandle) != NULL) {
		PIOS_USB_GET_DEVICE_INFO_HID_RES HidRes = (PIOS_USB_GET_DEVICE_INFO_HID_RES)
			HalIopAlloc(sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES));
		if (HidRes == NULL) {
			HalIopFree(Req);
			return STATUS_INSUFFICIENT_RESOURCES;
		}
		Res = HidRes;
		ResLength = sizeof(IOS_USB_GET_DEVICE_INFO_HID_RES);
		Body = (PIOS_USB_GET_DEVICE_INFO_RES_BODY)&HidRes->Device;
		Handle = s_hUsbHid;
		IsHid = TRUE;
	}
	
	RtlZeroMemory(Res, ResLength);
	NTSTATUS Status = HalIopIoctl(Handle, USB_IOCTL_GET_DEVICE_INFO, Req, sizeof(*Req), Res, ResLength, IOCTL_SWAP_NONE, IOCTL_SWAP_OUTPUT);
	
	// Don't need the request buffer any more.
	HalIopFree(Req);
	Req = NULL;
	
	if (!NT_SUCCESS(Status)) {
		HalIopFree(Res);
		return Status;
	}
	
	// Ensure everything unused is zeroed.
	RtlZeroMemory(Device, sizeof(*Device));
	RtlCopyMemory(&Device->Device, &Body->Device, sizeof(Device->Device));
	if (Device->Device.bNumConfigurations == 0) return Status;
	if (Device->Device.bNumConfigurations > 1) Device->Device.bNumConfigurations = 1;
	RtlCopyMemory(&Device->Config, &Body->Config, sizeof(Device->Config));
	// IOS claims each interface is its own device
	if (Device->Config.bNumInterfaces != 0) {
		Device->Config.bNumInterfaces = 1;
		RtlCopyMemory(&Device->Interface, &Body->Interface, sizeof(Device->Interface));
		if (IsHid) {
			// HID provides only an interface input and an interface output.
			// If either are not present they will be zeroed.
			UCHAR bNumEndpoints = 0;
			ULONG iEndpoint = 0;
			
			UCHAR dt = Body->Endpoints[iEndpoint].bDescriptorType;
			if (dt != 0) {
				RtlCopyMemory(&Device->Endpoints[bNumEndpoints], &Body->Endpoints[iEndpoint], sizeof(Device->Endpoints[0]));
				bNumEndpoints++;
			}
			iEndpoint++;
			
			dt = Body->Endpoints[iEndpoint].bDescriptorType;
			if (dt != 0) {
				RtlCopyMemory(&Device->Endpoints[bNumEndpoints], &Body->Endpoints[iEndpoint], sizeof(Device->Endpoints[0]));
				bNumEndpoints++;
			}
			Device->Interface.bNumEndpoints = bNumEndpoints;
		} else {
			// Skip vendor and class specific descriptors.
			ULONG iEndpoint = 0;
			for (iEndpoint = 0; iEndpoint < Device->Interface.bNumEndpoints; iEndpoint++) {
				UCHAR dt = Body->Endpoints[iEndpoint].bDescriptorType;
				if (dt == USB_DT_ENDPOINT || dt == USB_DT_INTERFACE) break;
			}
			Device->Interface.bNumEndpoints -= iEndpoint;
			RtlCopyMemory(Device->Endpoints, &Body->Endpoints[iEndpoint], sizeof(Device->Endpoints[0]) * Device->Interface.bNumEndpoints);
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
			pEndpoints = (PUSB_ENDPOINT)( (ULONG)pEndpoints + Interface->ExtraSize );
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
	
	HalIopFree(Res);
	return Status;
}

NTSTATUS UlGetGenericDescriptor(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR Type,
	UCHAR Index,
	UCHAR Interface,
	PVOID Data,
	ULONG Size
) {
	PVOID Buffer = HalIopAlloc(Size);
	if (Buffer == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	NTSTATUS Status = UlpGetDesc(DeviceHandle, Buffer, Type, Index, Interface, Size);
	if (NT_SUCCESS(Status)) {
		RtlCopyMemory(Data, Buffer, Size);
	}
	
	HalIopFree(Buffer);
	return Status;
}

NTSTATUS UlGetHidDescriptor(
	IOS_USB_HANDLE DeviceHandle,
	UCHAR Interface,
	PUSB_HID Hid,
	ULONG Size
) {
	if (Size < sizeof(USB_HID)) return STATUS_INVALID_PARAMETER;
	
	return UlGetGenericDescriptor(DeviceHandle, USB_DT_HID, 0, Interface, Hid, Size);
}

NTSTATUS UlGetReportDescriptorSize(IOS_USB_HANDLE DeviceHandle, UCHAR Interface, PUSHORT Length) {
	USB_HID Hid;
	NTSTATUS Status = UlGetHidDescriptor(DeviceHandle, Interface, &Hid, sizeof(Hid));
	if (!NT_SUCCESS(Status)) return Status;
	
	if (Hid.bLength > sizeof(Hid)) return STATUS_NO_SUCH_DEVICE;
	
	for (ULONG i = 0; i < Hid.bNumDescriptors; i++) {
		if (Hid.Descriptors[i].bDescriptorType == USB_DT_REPORT) {
			*Length = Hid.Descriptors[i].wDescriptorLength;
			return STATUS_SUCCESS;
		}
	}
	return STATUS_NO_SUCH_DEVICE;
}

NTSTATUS UlGetReportDescriptor(IOS_USB_HANDLE DeviceHandle, UCHAR Interface, PVOID Data, USHORT Length) {
	if (Data == NULL || Length < USB_DT_MINREPORT_SIZE) return STATUS_INVALID_PARAMETER;
	return UlGetGenericDescriptor(DeviceHandle, USB_DT_REPORT, 0, Interface, Data, Length);
}

#if 0
void UlFreeDescriptors(PUSB_DEVICE_DESC Device) {
	if (Device->Configs == NULL) {
		RtlZeroMemory(&Device, sizeof(*Device));
		return;
	}
	
	for (ULONG iConf = 0; iConf < Device->Desc.bNumConfigurations; iConf++) {
		PUSB_CONFIGURATION_DESC Config = &Device->Configs[iConf];
		if (Config->Desc.bNumInterfaces == 0) continue;
		PUSB_INTERFACE_DESC Interface = &Config->Interface;
		if (Interface->Extra != NULL) ExFreePool(Interface->Extra);
		if (Interface->Endpoints != NULL) ExFreePool(Interface->Endpoints);
	}
	ExFreePool(Device->Configs);
	RtlZeroMemory(&Device, sizeof(*Device));
}
#endif

NTSTATUS UlGetAsciiString(IOS_USB_HANDLE DeviceHandle, UCHAR Index, USHORT LangID, USHORT Length, PVOID Data, PUSHORT WrittenLength) {
	if (Length > USB_MAX_STRING_LENGTH) Length = USB_MAX_STRING_LENGTH;
	
	PUCHAR Buf = (PUCHAR)HalIopAlloc(USB_MAX_STRING_LENGTH);
	if (Buf == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	NTSTATUS Status = UlpGetDesc(DeviceHandle, Buf, USB_DT_STRING, Index, LangID, USB_MAX_STRING_LENGTH);
	
	if (NT_SUCCESS(Status)) {
		if (Index == 0) {
			// List of supported languages.
			RtlCopyMemory(Data, Buf, Length);
		} else {
			// Convert UTF-16LE to ASCII.
			UCHAR UnicodeIndex = 2;
			UCHAR AsciiIndex = 0;
			PUCHAR Data8 = (PUCHAR)Data;
			while (AsciiIndex < (Length - 1) && UnicodeIndex < Buf[0]) {
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
	HalIopFree(Buf);
	return Status;
}

NTSTATUS UlTransferIsoMessage(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, UCHAR Packets, PU16BE PacketSizes, PVOID Data) {
	return UlpSendIsoMessage(DeviceHandle, Endpoint, Packets, PacketSizes, Data, NULL, NULL, NULL);
}

NTSTATUS UlTransferIsoMessageAsync(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, UCHAR Packets, PU16BE PacketSizes, PVOID Data, PIOS_USB_ASYNC_RESULT Async, PVOID Context) {
	return UlpSendIsoMessage(DeviceHandle, Endpoint, Packets, PacketSizes, Data, Async, NULL, Context);
}

NTSTATUS UlTransferIsoMessageAsyncDpc(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, UCHAR Packets, PU16BE PacketSizes, PVOID Data, IOP_CALLBACK Callback, PVOID Context) {
	return UlpSendIsoMessage(DeviceHandle, Endpoint, Packets, PacketSizes, Data, NULL, Callback, Context);
}

NTSTATUS UlTransferInterruptMessage(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data) {
	return UlpSendInterruptMessage(DeviceHandle, Endpoint, Length, Data, NULL, NULL, NULL);
}

NTSTATUS UlTransferInterruptMessageAsync(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data, PIOS_USB_ASYNC_RESULT Async, PVOID Context) {
	return UlpSendInterruptMessage(DeviceHandle, Endpoint, Length, Data, Async, NULL, Context);
}

NTSTATUS UlTransferInterruptMessageAsyncDpc(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data, IOP_CALLBACK Callback, PVOID Context) {
	return UlpSendInterruptMessage(DeviceHandle, Endpoint, Length, Data, NULL, Callback, Context);
}

NTSTATUS UlTransferBulkMessage(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data) {
	return UlpSendBulkMessage(DeviceHandle, Endpoint, Length, Data, NULL, NULL, NULL);
}

NTSTATUS UlTransferBulkMessageAsync(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data, PIOS_USB_ASYNC_RESULT Async, PVOID Context) {
	return UlpSendBulkMessage(DeviceHandle, Endpoint, Length, Data, Async, NULL, Context);
}

NTSTATUS UlTransferBulkMessageAsyncDpc(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint, USHORT Length, PVOID Data, IOP_CALLBACK Callback, PVOID Context) {
	return UlpSendBulkMessage(DeviceHandle, Endpoint, Length, Data, NULL, Callback, Context);
}

NTSTATUS UlTransferControlMessage(IOS_USB_HANDLE DeviceHandle, UCHAR RequestType, UCHAR Request, USHORT Value, USHORT Index, USHORT Length, PVOID Data) {
	return UlpSendControlMessage(DeviceHandle, RequestType, Request, Value, Index, Length, Data, NULL, NULL, NULL);
}

NTSTATUS UlTransferControlMessageAsync(IOS_USB_HANDLE DeviceHandle, UCHAR RequestType, UCHAR Request, USHORT Value, USHORT Index, USHORT Length, PVOID Data, PIOS_USB_ASYNC_RESULT Async, PVOID Context) {
	return UlpSendControlMessage(DeviceHandle, RequestType, Request, Value, Index, Length, Data, Async, NULL, Context);
}

NTSTATUS UlTransferControlMessageAsyncDpc(IOS_USB_HANDLE DeviceHandle, UCHAR RequestType, UCHAR Request, USHORT Value, USHORT Index, USHORT Length, PVOID Data, IOP_CALLBACK Callback, PVOID Context) {
	return UlpSendControlMessage(DeviceHandle, RequestType, Request, Value, Index, Length, Data, NULL, Callback, Context);
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
			RtlCopyMemory(&Entry[Wrote], &LocalEntry, sizeof(LocalEntry));
			Wrote++;
			i++;
		}
	}
	if (InterfaceClass == 0 || InterfaceClass == USB_CLASS_HID) {
		ULONG i = 0;
		while (i < USB_COUNT_DEVICES && Wrote < Count && s_AttachedDevicesHid->Entries[i].DeviceHandle != 0) {
			// Pointer could be in uncached DDR, write to stack then copy
			LocalEntry.DeviceHandle = NativeReadBase32(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].DeviceHandle));
			LocalEntry.VendorId = NativeReadBase16(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].VendorId));
			LocalEntry.ProductId = NativeReadBase16(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].ProductId));
			LocalEntry.Token = NativeReadBase32(MMIO_OFFSET(s_AttachedDevicesHid, Entries[i].Token));
			RtlCopyMemory(&Entry[Wrote], &LocalEntry, sizeof(LocalEntry));
			Wrote++;
			i++;
		}
	}
	if (WrittenCount != NULL) *WrittenCount = Wrote;
}

NTSTATUS UlSetConfiguration(IOS_USB_HANDLE DeviceHandle, UCHAR Configuration) {
	return UlpSendControlMessage(
		DeviceHandle,
		(USB_CTRLTYPE_DIR_HOST2DEVICE | USB_CTRLTYPE_TYPE_STANDARD | USB_CTRLTYPE_REC_DEVICE),
		USB_REQ_SETCONFIG,
		Configuration,
		0,
		0,
		NULL,
		NULL,
		NULL,
		NULL
	);
}

NTSTATUS UlGetConfiguration(IOS_USB_HANDLE DeviceHandle, PUCHAR Configuration) {
	PUCHAR Buffer = (PUCHAR)HalIopAlloc(sizeof(UCHAR));
	if (Buffer == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	NTSTATUS Status = UlpSendControlMessage(
		DeviceHandle, 
		(USB_CTRLTYPE_DIR_DEVICE2HOST | USB_CTRLTYPE_TYPE_STANDARD | USB_CTRLTYPE_REC_DEVICE),
		USB_REQ_GETCONFIG,
		0,
		0,
		sizeof(UCHAR),
		Buffer,
		NULL,
		NULL,
		NULL
	);
	if (NT_SUCCESS(Status)) *Configuration = *Buffer;
	HalIopFree(Buffer);
	return Status;
}

NTSTATUS UlSetAlternativeInterface(IOS_USB_HANDLE DeviceHandle, UCHAR Interface, UCHAR AlternateSetting) {
	return UlpSendControlMessage(
		DeviceHandle,
		(USB_CTRLTYPE_DIR_HOST2DEVICE | USB_CTRLTYPE_TYPE_STANDARD | USB_CTRLTYPE_REC_INTERFACE),
		USB_REQ_SETINTERFACE,
		AlternateSetting,
		Interface,
		0,
		NULL,
		NULL,
		NULL,
		NULL
	);
}

static NTSTATUS UlpCancelEndpoint(IOS_HANDLE Handle, IOS_USB_HANDLE DeviceHandle, PIOS_USB_CANCEL_ENDPOINT_REQ Req, UCHAR Endpoint) {
	STACK_ALIGN(IOS_USB_CANCEL_ENDPOINT_REQ, pReq, 1, 8);
	RtlZeroMemory(pReq, sizeof(*pReq));
	NATIVE_WRITE(pReq, DeviceHandle, DeviceHandle);
	NATIVE_WRITE(pReq, Endpoint, Endpoint);
	RtlCopyMemory(Req, pReq, sizeof(*pReq));
	return HalIopIoctl(Handle, USB_IOCTL_CANCEL_ENDPOINT, Req, sizeof(*Req), NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
}

NTSTATUS UlCancelEndpoint(IOS_USB_HANDLE DeviceHandle, UCHAR Endpoint) {
	PIOS_USB_CANCEL_ENDPOINT_REQ Req = (PIOS_USB_CANCEL_ENDPOINT_REQ)
		HalIopAlloc(sizeof(IOS_USB_CANCEL_ENDPOINT_REQ));
	if (Req == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	IOS_HANDLE Handle = IOS_HANDLE_INVALID;
	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		Handle = s_hUsbVen;
	} else if (UlpFindDeviceHid(DeviceHandle) != NULL) {
		Handle = s_hUsbHid;
	}
	
	NTSTATUS Status = STATUS_NO_SUCH_DEVICE;
	do {
		if (Handle == IOS_HANDLE_INVALID) break;
		Status = UlpCancelEndpoint(Handle, DeviceHandle, Req, Endpoint);
	} while (0);
	
	HalIopFree(Req);
	return Status;
}

NTSTATUS UlClearHalt(IOS_USB_HANDLE DeviceHandle) {
	PIOS_USB_CANCEL_ENDPOINT_REQ Req = (PIOS_USB_CANCEL_ENDPOINT_REQ)
		HalIopAlloc(sizeof(IOS_USB_CANCEL_ENDPOINT_REQ));
	if (Req == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	
	IOS_HANDLE Handle = IOS_HANDLE_INVALID;
	if (UlpFindDeviceVen(DeviceHandle) != NULL) {
		Handle = s_hUsbVen;
	} else if (UlpFindDeviceHid(DeviceHandle) != NULL) {
		Handle = s_hUsbHid;
	}
	
	NTSTATUS Status = STATUS_NO_SUCH_DEVICE;
	do {
		if (Handle == IOS_HANDLE_INVALID) break;
		// Cancel control messages
		Status = UlpCancelEndpoint(Handle, DeviceHandle, Req, USB_CANCEL_CONTROL);
		if (!NT_SUCCESS(Status)) break;
		// Cancel incoming messages
		Status = UlpCancelEndpoint(Handle, DeviceHandle, Req, USB_CANCEL_INCOMING);
		if (!NT_SUCCESS(Status)) break;
		// Cancel outgoing messages
		Status = UlpCancelEndpoint(Handle, DeviceHandle, Req, USB_CANCEL_OUTGOING);
	} while (0);
	
	HalIopFree(Req);
	return Status;
}

PVOID UlGetPassedAsyncContext(PVOID AsyncContext) {
	if (AsyncContext == NULL) return NULL;
	PIOS_USB_CONTROL_TRANSFER_REQ Req = (PIOS_USB_CONTROL_TRANSFER_REQ)
		AsyncContext;
	PVOID Ret = Req->Context;
	
	HalIopFree(Req);
	return Ret;
}