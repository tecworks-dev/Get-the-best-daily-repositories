#pragma once

// IOS file/device handle.
typedef ULONG IOS_HANDLE, *PIOS_HANDLE;
enum {
	IOS_HANDLE_INVALID = 0xffffffff
};

// IOS open mode bitflags (read/write)
typedef enum _IOS_OPEN_MODE {
	IOSOPEN_NONE = 0,
	IOSOPEN_READ = BIT(0),
	IOSOPEN_WRITE = BIT(1)
} IOS_OPEN_MODE;

// IOS seek mode
typedef enum _IOS_SEEK_MODE {
	IOSSEEK_SET,
	IOSSEEK_CURRENT,
	IOSSEEK_END
} IOS_SEEK_MODE;

// Describes a buffer sent to a vectored IOCTL handler.
typedef struct _IOS_IOCTL_VECTOR {
	ULONG Length;
	PVOID Pointer;
} IOS_IOCTL_VECTOR, *PIOS_IOCTL_VECTOR;

// IOS read/write mode bitflags (buffer swap)
typedef enum _IOS_RW_MODE {
	RW_SWAP_NONE = 0, // Do not endianness swap the buffer.
	RW_SWAP_INPUT = BIT(0), // Endianness swap the buffer before ioctl call.
	RW_SWAP_OUTPUT = BIT(1), // Endianness swap the buffer after ioctl response.
	RW_SWAP_BOTH = RW_SWAP_INPUT | RW_SWAP_OUTPUT // Endianness swap buffer both before call and after response.
} IOS_RW_MODE;

// IOS ioctl mode bitflags (buffer swap)
typedef enum _IOS_IOCTL_MODE {
	IOCTL_SWAP_NONE = 0, // Do not endianness swap either buffer.
	IOCTL_SWAP_INPUT = BIT(0), // Endianness swap the input buffer.
	IOCTL_SWAP_OUTPUT = BIT(1), // Endianness swap the output buffer.
	IOCTL_SWAP_BOTH = IOCTL_SWAP_INPUT | IOCTL_SWAP_OUTPUT, // Endianness swap both buffers.
} IOS_IOCTL_MODE;

// Allocate Size bytes from the 128KB PXI area in DDR.
NTHALAPI PVOID HalIopAlloc(ULONG Size);

// Allocate IOCTL vectors from PXI RAM.
#define HalIopVectorAlloc(Count) (PIOS_IOCTL_VECTOR) \
	HalIopAlloc( (Count) * sizeof(IOS_IOCTL_VECTOR) )

// Free a buffer allocated using HalIopAlloc.
NTHALAPI void HalIopFree(PVOID Buffer);

// Base status code for IOS successful result.
#define STATUS_IOP_SUCCESS 0x1000

// Convert status code to successful IOS result.
#define STATUS_TO_IOP(Status) (LONG)( (Status) == 0 ? 0 : (Status) - STATUS_IOP_SUCCESS )

// IOS asynchronous API for NT:
// provide a pointer to NTSTATUS,
// and a pointer to an event that is raised on operation completion
// when the event is raised, the NTSTATUS will have been written.
// for IOS_Open, the IOS_HANDLE will also have been written at that time.

// Additionally, an async variant is provided with a callback called in DPC.

// All passed in pointers must be in nonpageable memory.
// If any are not, functions will return STATUS_INVALID_PARAMETER.

// Callback.
typedef void (*IOP_CALLBACK)(NTSTATUS Status, ULONG Result, PVOID Context);

// Opens an IOS filesystem path (synchronous)
NTHALAPI NTSTATUS HalIopOpen(const char * Path, IOS_OPEN_MODE Mode, IOS_HANDLE* Handle);

// Opens an IOS filesystem path (asynchronous)
NTHALAPI NTSTATUS HalIopOpenAsync(const char * Path, IOS_OPEN_MODE Mode, IOS_HANDLE* Handle, NTSTATUS* pStatus, PRKEVENT Event);

// Opens an IOS filesystem path (asynchronous DPC)
NTHALAPI NTSTATUS HalIopOpenAsyncDpc(const char * Path, IOS_OPEN_MODE Mode, IOP_CALLBACK Callback, PVOID Context);

// Closes an IOS handle. (synchronous)
NTHALAPI NTSTATUS HalIopClose(IOS_HANDLE Handle);

// Closes an IOS handle. (asynchronous)
NTHALAPI NTSTATUS HalIopCloseAsync(IOS_HANDLE Handle, NTSTATUS* pStatus, PRKEVENT Event);

// Closes an IOS handle. (asynchronous DPC)
NTHALAPI NTSTATUS HalIopCloseAsyncDpc(IOS_HANDLE Handle, IOP_CALLBACK Callback, PVOID Context);

// Reads from an IOS handle. (synchronous)
NTHALAPI NTSTATUS HalIopRead(IOS_HANDLE Handle, PVOID Buffer, ULONG Length, IOS_RW_MODE SwapMode, PULONG NumberOfBytesTransferred);

// Reads from an IOS handle. (asynchronous)
NTHALAPI NTSTATUS HalIopReadAsync(IOS_HANDLE Handle, PVOID Buffer, ULONG Length, IOS_RW_MODE SwapMode, PULONG NumberOfBytesTransferred, NTSTATUS* pStatus, PRKEVENT Event);

// Reads from an IOS handle. (asynchronous DPC)
NTHALAPI NTSTATUS HalIopReadAsyncDpc(IOS_HANDLE Handle, PVOID Buffer, ULONG Length, IOS_RW_MODE SwapMode, IOP_CALLBACK Callback, PVOID Context);

// Writes to an IOS handle. (synchronous)
NTHALAPI NTSTATUS HalIopWrite(IOS_HANDLE Handle, PVOID Buffer, ULONG Length, IOS_RW_MODE SwapMode, PULONG NumberOfBytesTransferred);

// Writes to an IOS handle. (asynchronous)
NTHALAPI NTSTATUS HalIopWriteAsync(IOS_HANDLE Handle, PVOID Buffer, ULONG Length, IOS_RW_MODE SwapMode, PULONG NumberOfBytesTransferred, NTSTATUS* pStatus, PRKEVENT Event);

// Writes to an IOS handle. (asynchronous DPC)
NTHALAPI NTSTATUS HalIopWriteAsyncDpc(IOS_HANDLE Handle, PVOID Buffer, ULONG Length, IOS_RW_MODE SwapMode, IOP_CALLBACK Callback, PVOID Context);

// Seeks to a new file offset. (synchronous)
NTHALAPI NTSTATUS HalIopSeek(IOS_HANDLE Handle, LONG Offset, IOS_SEEK_MODE Mode, PULONG FileOffset);

// Seeks to a new file offset. (asynchronous)
NTHALAPI NTSTATUS HalIopSeekAsync(IOS_HANDLE Handle, LONG Offset, IOS_SEEK_MODE Mode, PULONG FileOffset, NTSTATUS* pStatus, PRKEVENT Event);

// Seeks to a new file offset. (asynchronous DPC)
NTHALAPI NTSTATUS HalIopSeekAsyncDpc(IOS_HANDLE Handle, LONG Offset, IOS_SEEK_MODE Mode, IOP_CALLBACK Callback, PVOID Context);

// Sends a non-vectored device-specific request to an IOS handle. (synchronous)
NTHALAPI NTSTATUS HalIopIoctl(IOS_HANDLE Handle, ULONG ControlCode, PVOID Input, ULONG LengthInput, PVOID Output, ULONG LengthOutput, IOS_IOCTL_MODE SwapModeInput, IOS_IOCTL_MODE SwapModeOutput);

// Sends a non-vectored device-specific request to an IOS handle. (asynchronous)
NTHALAPI NTSTATUS HalIopIoctlAsync(IOS_HANDLE Handle, ULONG ControlCode, PVOID Input, ULONG LengthInput, PVOID Output, ULONG LengthOutput, IOS_IOCTL_MODE SwapModeInput, IOS_IOCTL_MODE SwapModeOutput, NTSTATUS* pStatus, PRKEVENT Event);

// Sends a non-vectored device-specific request to an IOS handle. (asynchronous DPC)
NTHALAPI NTSTATUS HalIopIoctlAsyncDpc(IOS_HANDLE Handle, ULONG ControlCode, PVOID Input, ULONG LengthInput, PVOID Output, ULONG LengthOutput, IOS_IOCTL_MODE SwapModeInput, IOS_IOCTL_MODE SwapModeOutput, IOP_CALLBACK Callback, PVOID Context);

// Sends a vectored device-specific request to an IOS handle. (synchronous)
NTHALAPI NTSTATUS HalIopIoctlv(IOS_HANDLE Handle, ULONG ControlCode, ULONG NumRead, ULONG NumWritten, PIOS_IOCTL_VECTOR Buffers, ULONG SwapBuffersIn, ULONG SwapBuffersOut);

// Sends a vectored device-specific request to an IOS handle. (asynchronous)
NTHALAPI NTSTATUS HalIopIoctlvAsync(IOS_HANDLE Handle, ULONG ControlCode, ULONG NumRead, ULONG NumWritten, PIOS_IOCTL_VECTOR Buffers, ULONG SwapBuffersIn, ULONG SwapBuffersOut, NTSTATUS* pStatus, PRKEVENT Event);

// Sends a vectored device-specific request to an IOS handle. (asynchronous DPC)
NTHALAPI NTSTATUS HalIopIoctlvAsyncDpc(IOS_HANDLE Handle, ULONG ControlCode, ULONG NumRead, ULONG NumWritten, PIOS_IOCTL_VECTOR Buffers, ULONG SwapBuffersIn, ULONG SwapBuffersOut, IOP_CALLBACK Callback, PVOID Context);
