#pragma once
#include "exiapi.h"

// EXI hardware registers.

typedef union _EXI_CHANNEL_PARAMETER_REGISTER {
	struct {
		BOOLEAN DeviceInterruptMask : 1;
		BOOLEAN DeviceInterruptStatus : 1;
		BOOLEAN TransferInterruptMask : 1;
		BOOLEAN TransferInterruptStatus : 1;
		EXI_CLOCK_FREQUENCY ClockFrequency : 3;
		BOOLEAN ChipSelect0 : 1;
		
		BOOLEAN ChipSelect1 : 1;
		BOOLEAN ChipSelect2 : 1;
		BOOLEAN UnplugInterruptMask : 1;
		BOOLEAN UnplugInterruptStatus : 1;
		BOOLEAN Connected : 1;
		BOOLEAN BootromDisable : 1;
	};
	ULONG Value;
} EXI_CHANNEL_PARAMETER_REGISTER, *PEXI_CHANNEL_PARAMETER_REGISTER;

typedef union _EXI_CHANNEL_TRANSFER_REGISTER {
	struct {
		BOOLEAN Start : 1;
		BOOLEAN EnableDma : 1;
		EXI_TRANSFER_TYPE Type : 2;
		UCHAR Length : 2; // zero = 1 byte, etc.
	};
	ULONG Value;
} EXI_CHANNEL_TRANSFER_REGISTER, *PEXI_CHANNEL_TRANSFER_REGISTER;

typedef struct _EXI_CHANNEL_REGISTERS {
	EXI_CHANNEL_PARAMETER_REGISTER Parameter;
	ULONG DmaAddress;
	ULONG DmaLength;
	EXI_CHANNEL_TRANSFER_REGISTER Transfer;
	ULONG Data;
} EXI_CHANNEL_REGISTERS, *PEXI_CHANNEL_REGISTERS;

#if 0
typedef struct {
	volatile ULONG Status;
	volatile ULONG Address;
	volatile ULONG Length;
	volatile ULONG Control;
	volatile ULONG Data;
} EXI_CHANNEL_REGISTERS, *PEXI_CHANNEL_REGISTERS;
#endif

typedef struct {
	EXI_CHANNEL_REGISTERS Channel[3];
	ULONG Reserved;
	ULONG BootVector[(0x80 - 0x40) / 4];
} EXI_REGISTERS, *PEXI_REGISTERS;

enum {
	EXI_REGISTER_BASE_FLIPPER = 0x0C006800,
	EXI_REGISTER_BASE = 0x0D006800
};