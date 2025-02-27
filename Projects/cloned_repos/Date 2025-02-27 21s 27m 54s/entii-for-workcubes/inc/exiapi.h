#pragma once
#include "runtime.h"

enum {
	EXI_CHANNEL_COUNT = 3,
	EXI_DEVICE_COUNT = 3
};

typedef enum {
	EXI_CLOCK_0_8, // 0.84375MHz
	EXI_CLOCK_1_6, // 1.6875MHz
	EXI_CLOCK_3_3, // 3.375MHz
	EXI_CLOCK_6_7, // 6.75MHz
	EXI_CLOCK_13_5, // 13.5MHz
	EXI_CLOCK_27, // 27MHz
	EXI_CLOCK_54, // 54MHz (Vegas and Latte only)
} EXI_CLOCK_FREQUENCY;

typedef enum {
	EXI_TRANSFER_READ,
	EXI_TRANSFER_WRITE,
	EXI_TRANSFER_READWRITE
} EXI_TRANSFER_TYPE;

// DMA swap mode
typedef enum {
	EXI_SWAP_NONE = 0, // Do not endianness swap the buffer.
	EXI_SWAP_INPUT = BIT(0), // Endianness swap the buffer before transfer.
	EXI_SWAP_OUTPUT = BIT(1), // Endianness swap the buffer after transfer.
	EXI_SWAP_BOTH = EXI_SWAP_INPUT | EXI_SWAP_OUTPUT // Endianness swap buffer both before and after transfer.
} EXI_SWAP_MODE;

enum {
	EXI_CHAN0_DEVICE_MEMCARD = 0, // Memory Card #1
	EXI_CHAN0_DEVICE_RTC = 1, // Macronix SPI 16MBit ROM with RTC
	EXI_CHAN0_DEVICE_SP1 = 2, // Serial Port 1 (on flipper systems only)

	EXI_CHAN1_DEVICE_MEMCARD1 = 0, // Memory Card #2

	EXI_CHAN2_DEVICE_SP2 = 0, // Serial Port 2 (on flipper systems only)
};

typedef enum {
	ExiUnlock = 0, // EXI channel will be unlocked upon return from this function (only synchronous transfers were performed)
	ExiKeepLocked = 1 // EXI channel will remain locked upon return from this function (async transfers were started)
} EXI_LOCK_ACTION;

typedef enum {
	ExiInterruptDisable = 0, // EXI channel device interrupt remains disabled
	ExiInterruptEnable = 1 // EXI channel device interrupt is re-enabled (ie, callback successfully cleared the device interrupt)
} EXI_INTERRUPT_ACTION;

// Called when EXI device interrupt is raised.
// Arguments:
// channel = EXI channel (if 0, then this is for memcard device)
// isLocked = if FALSE, then the exi channel is NOT locked; so EXI transfers cannot occur, and ExiInterruptDisable is only valid return value.
// wasLocked = if FALSE, then the exi channel will be unlocked after return from this function, so asynchronous transfers cannot occur
typedef EXI_INTERRUPT_ACTION (*HAL_EXI_INTERRUPT_CALLBACK)(ULONG channel, BOOLEAN isLocked, BOOLEAN wasLocked);
// Called when EXI channel has been locked.
// Arguments:
// channel = EXI channel
// context = context passed to HalExiLock
typedef EXI_LOCK_ACTION (*HAL_EXI_LOCK_CALLBACK)(ULONG channel, PVOID context);
typedef EXI_LOCK_ACTION (*HAL_EXI_IMMASYNC_CALLBACK)(ULONG channel, ULONG data, PVOID context);
typedef EXI_LOCK_ACTION (*HAL_EXI_ASYNC_CALLBACK)(ULONG channel, PVOID context);

// In this implementation, HalExiTransferDma just works (falling back to immediate if needed):
#define HalExiBufferCanDma(buffer, length) TRUE

NTHALAPI BOOLEAN HalExiIsDevicePresent(ULONG channel, ULONG device);
NTHALAPI NTSTATUS HalExiLock(ULONG channel, HAL_EXI_LOCK_CALLBACK callback, PVOID context);
NTHALAPI NTSTATUS HalExiLockNonpaged(ULONG channel, HAL_EXI_LOCK_CALLBACK callback, PVOID context);

// all functions below require a locked channel

NTHALAPI NTSTATUS HalExiSelectDevice(ULONG channel, ULONG device, EXI_CLOCK_FREQUENCY frequency, BOOLEAN CsHigh);
NTHALAPI NTSTATUS HalExiUnselectDevice(ULONG channel);
NTHALAPI NTSTATUS HalExiRefreshDevice(ULONG channel);
NTHALAPI NTSTATUS HalExiTransferImmediate(ULONG channel, ULONG data, ULONG length, EXI_TRANSFER_TYPE type, PULONG dataRead);
NTHALAPI NTSTATUS HalExiTransferImmediateBuffer(ULONG channel, PVOID bufferRead, PVOID bufferWrite, ULONG length, EXI_TRANSFER_TYPE type);
NTHALAPI NTSTATUS HalExiReadWriteImmediateOutBuffer(ULONG channel, UCHAR byteRead, PVOID buffer, ULONG length);
NTHALAPI NTSTATUS HalExiTransferDma(ULONG channel, PVOID buffer, ULONG length, EXI_TRANSFER_TYPE type, EXI_SWAP_MODE swap);
NTHALAPI NTSTATUS HalExiGetDeviceIdentifier(ULONG channel, ULONG device, PULONG deviceIdentifier);

// Should be called from somewhere else (callback?) after a lock callback has executed with return value ExiKeepLocked
NTHALAPI NTSTATUS HalExiUnlock(ULONG channel);
NTHALAPI NTSTATUS HalExiUnlockNonpaged(ULONG channel);

// All async methods must ONLY be passed pointers to nonpaged memory
NTHALAPI NTSTATUS HalExiTransferImmediateAsync(ULONG channel, ULONG data, ULONG length, EXI_TRANSFER_TYPE type, HAL_EXI_IMMASYNC_CALLBACK callback, PVOID context);
NTHALAPI NTSTATUS HalExiTransferImmediateBufferAsync(ULONG channel, PVOID bufferRead, PVOID bufferWrite, ULONG length, EXI_TRANSFER_TYPE type, HAL_EXI_ASYNC_CALLBACK callback, PVOID context);
NTHALAPI NTSTATUS HalExiReadWriteImmediateOutBufferAsync(ULONG channel, UCHAR byteRead, PVOID buffer, ULONG length, HAL_EXI_ASYNC_CALLBACK callback, PVOID context);
NTHALAPI NTSTATUS HalExiTransferDmaAsync(ULONG channel, PVOID buffer, ULONG length, EXI_TRANSFER_TYPE type, EXI_SWAP_MODE swap, HAL_EXI_ASYNC_CALLBACK callback, PVOID context);

NTHALAPI NTSTATUS HalExiToggleInterrupt(ULONG channel, HAL_EXI_INTERRUPT_CALLBACK callback);