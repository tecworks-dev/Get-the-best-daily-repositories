#pragma once
#include "types.h"
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

void ExiInit(ARTX_SYSTEM_TYPE SystemType);
bool ExiIsDevicePresent(ULONG channel, ULONG device);
bool ExiSelectDevice(ULONG channel, ULONG device, EXI_CLOCK_FREQUENCY frequency, bool CsHigh);
bool ExiUnselectDevice(ULONG channel);
bool ExiRefreshDevice(ULONG channel);
bool ExiTransferImmediate(ULONG channel, ULONG data, ULONG length, EXI_TRANSFER_TYPE type, PULONG dataRead);
bool ExiTransferImmediateBuffer(ULONG channel, PVOID bufferRead, PVOID bufferWrite, ULONG length, EXI_TRANSFER_TYPE type);
bool ExiReadWriteImmediateOutBuffer(ULONG channel, UCHAR byteRead, PVOID buffer, ULONG length);
bool ExiTransferDma(ULONG channel, PVOID buffer, ULONG length, EXI_TRANSFER_TYPE type, EXI_SWAP_MODE swap);
bool ExiReadDmaWithImmediate(ULONG channel, UCHAR immediate, PVOID buffer, ULONG length, EXI_SWAP_MODE swap);
bool ExiBufferCanDma(PVOID buffer, ULONG length);
bool ExiGetDeviceIdentifier(ULONG channel, ULONG device, PULONG deviceIdentifier);
ULONG ExiReadDataRegister(ULONG channel);