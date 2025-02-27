#pragma once
#include "iosapi.h"

typedef enum _IOS_OPERATION {
	IOS_OPEN = 1,
	IOS_CLOSE,
	IOS_READ,
	IOS_WRITE,
	IOS_SEEK,
	IOS_IOCTL,
	IOS_IOCTLV,
	IOS_ASYNC_RESPONSE,
	__IOS_OPERATION_U32_MARKER = 0xFFFFFFFF
} IOS_OPERATION;

// Memory as PPC(MSR_LE) sees it.
typedef struct _IOS_IPC_OPEN {
	union {
		ULONG Name;
		PCHAR _le_Name;
	};
	ULONG Handle;
	ULONG Padding;
	IOS_OPEN_MODE Mode;
} IOS_IPC_OPEN, *PIOS_IPC_OPEN;

typedef struct _IOS_IPC_BUFFER {
	ULONG Length;
	union {
		ULONG Pointer;
		PVOID _le_Pointer;
	};
} IOS_IPC_BUFFER, *PIOS_IPC_BUFFER;

typedef struct _IOS_IPC_RW {
	ULONG Length;
	ULONG Handle;
	ULONG Padding;
	union {
		ULONG Pointer;
		PVOID _le_Pointer;
	};
} IOS_IPC_RW, *PIOS_IPC_RW;

typedef struct _IOS_IPC_SEEK {
	LONG Offset;
	ULONG Handle;
	ULONG Padding;
	IOS_SEEK_MODE Mode;
} IOS_IPC_SEEK, *PIOS_IPC_SEEK;

typedef struct _IOS_IPC_IOCTL {
	ULONG ControlCode;
	ULONG Handle;
	IOS_IPC_BUFFER Input;
	IOS_IPC_BUFFER Output;
} IOS_IPC_IOCTL, *PIOS_IPC_IOCTL;

typedef struct _IOS_IPC_IOCTLV {
	ULONG ControlCode;
	ULONG Handle;
	ULONG NumWritten;
	ULONG NumRead;
	ULONG Padding;
	union {
		ULONG Buffers;
		PIOS_IPC_BUFFER _le_Buffers;
	};
} IOS_IPC_IOCTLV, *PIOS_IPC_IOCTLV;

typedef union _IOS_IPC_ARGS {
	IOS_IPC_OPEN Open;
	struct { } Close;
	IOS_IPC_RW Read, Write;
	IOS_IPC_SEEK Seek;
	IOS_IPC_IOCTL Ioctl;
	IOS_IPC_IOCTLV Ioctlv;
} IOS_IPC_ARGS, *PIOS_IPC_ARGS;

typedef struct _IOS_IPC_DESC {
	LONG Result;
	IOS_OPERATION Operation;
	union {
		IOS_IPC_ARGS Args;
		struct {
			ULONG IpcArg0;
			union {
				IOS_HANDLE Handle;
				IOS_OPERATION ReqOp;
			};
		};
	};
} IOS_IPC_DESC, *PIOS_IPC_DESC;

_Static_assert(sizeof(IOS_IPC_DESC) == 0x20);

typedef struct _IOS_IPC_REQUEST {
	IOS_IPC_DESC Ipc; // The IPC request itself
	IOS_IPC_DESC IpcVirt; // The same IPC request but using virtual addresses.
	ULONG Flags;
	union {
		PRKEVENT Event; // Event handle to raise when request has responded
		IOP_CALLBACK Callback; // Callback to be called at DPC level.
	};
	NTSTATUS* Status; // Pointer to where the NTSTATUS code is that is returned to the caller.
	union {
		PIOS_HANDLE Handle; // Pointer to the returned IOS handle (for IOS_OPEN)
		PULONG NumberOfBytesTransferred; // Pointer to the returned number of bytes transferred (for IOS_READ, IOS_WRITE)
		PULONG FileOffset; // Pointer to the new file offset (for IOS_SEEK)
		PVOID Context; // DPC callback context.
	};
	ULONG SwapModeInput; // Swap mode bits for input buffers. For ioctlv, single bit for each buffer. For r/w/ioctl, swap mode bits (both enums are same)
	ULONG SwapModeOutput; // Swap mode bits for output buffers. For ioctlv, single bit for each buffer. For r/w/ioctl, swap mode bits (both enums are same)
	ULONG Padding[2]; // Extra space for struct length to correctly align
} IOS_IPC_REQUEST, *PIOS_IPC_REQUEST;

_Static_assert((sizeof(IOS_IPC_DESC) & 0xF) == 0);

typedef struct _IOS_IPC_REQUEST_STATIC {
	IOS_IPC_REQUEST Request; // The base IPC request buffer
	IOS_IPC_BUFFER Ioctlv[8]; // Ioctlv pointer space for up to 8 args
} IOS_IPC_REQUEST_STATIC, *PIOS_IPC_REQUEST_STATIC;

enum {
	IPC_FLAG_CALLBACK = BIT(0), // Callback is used here.
	IPC_FLAG_STATIC = BIT(1), // This is a static buffer.
	IPC_FLAG_SYNC = BIT(2), // This is a synchronous request.
};

typedef struct _IOS_IPC_SYNC {
	KEVENT Event;
	NTSTATUS Status;
} IOS_IPC_SYNC, *PIOS_IPC_SYNC;

