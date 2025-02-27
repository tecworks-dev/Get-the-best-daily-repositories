// Dolphin External Interface. SPI-like bus with DMA (although can be used with SPI devices if registers are configured properly). Used for the memory card slots and the bottom connector on gamecube only.
// Base address is at 0x0C006800 on flipper, 0x0D006800 on vegas and latte.
// Dolphin (emulator) sets up both areas when emulating broadway + parts of vegas, but vegas real hw only allows for one or the other mapping to be used (and with IOS drivers running, we need the second)

#include <stdio.h>
#include <memory.h>
#include "arc.h"
#include "types.h"
#include "runtime.h"
#include "arcmem.h"
#include "exi.h"
#include "exi_sdmc.h"

// EXI register definitions.

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

typedef struct _EXI_REGISTERS {
	EXI_CHANNEL_REGISTERS Channel[EXI_CHANNEL_COUNT];
	ULONG Reserved;
	ULONG BootVector[(0x80 - 0x40) / 4];
} EXI_REGISTERS, *PEXI_REGISTERS;
_Static_assert(sizeof(EXI_REGISTERS) == 0x80);

enum {
	EXI_REGISTER_BASE_FLIPPER = 0x0C006800,
	EXI_REGISTER_OFFSET_VEGAS = (0x0D006800 - EXI_REGISTER_BASE_FLIPPER)
};

static __attribute__((const)) inline ULONG ExipGetPhysBase(ARTX_SYSTEM_TYPE SystemType) {
	ULONG base = EXI_REGISTER_BASE_FLIPPER;
	if (SystemType != ARTX_SYSTEM_FLIPPER) base += EXI_REGISTER_OFFSET_VEGAS;
	return base;
}

static int64_t ExipBitmaskTest(void);

static PEXI_REGISTERS ExiRegisters;

#define EXI_CLOCK_ZERO EXI_CLOCK_0_8

static bool ExipDeviceSelected(ULONG channel) {
	// Ensure a chip select line is pulled low, or frequency is non-zero
	EXI_CHANNEL_PARAMETER_REGISTER Cpr;
	Cpr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Parameter));
	if (Cpr.ChipSelect0 == 0 && Cpr.ChipSelect1 == 0 && Cpr.ChipSelect2 == 0) {
		if (Cpr.ClockFrequency == EXI_CLOCK_ZERO) return false; // No device selected.
	}

	return true;
}

bool ExiIsDevicePresent(ULONG channel, ULONG device) {
	if (channel >= EXI_CHANNEL_COUNT) return false;
	if (device >= EXI_DEVICE_COUNT) return false;
	// for channel 0, this only detects memcard.
	if (channel == 0 && device != EXI_CHAN0_DEVICE_MEMCARD) return true;
	// for channel 1 and 2, only device 0 exists
	if (channel != 0 && device != 0) return false;
	// EXI2EXTIN pin does not exist on any system, so always assume a device is present on channel 2
	if (channel == 2) return true;

	EXI_CHANNEL_PARAMETER_REGISTER Cpr;
	Cpr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Parameter));

	return Cpr.Connected;
}

bool ExiSelectDevice(ULONG channel, ULONG device, EXI_CLOCK_FREQUENCY frequency, bool CsHigh) {
	if (channel >= EXI_CHANNEL_COUNT) return false;
	if (device >= EXI_DEVICE_COUNT) return false;
	if (channel != 0 && device != 0) return false;
	if ((ULONG)frequency > EXI_CLOCK_54) return false;
	// Do not allow high chip select with 0.8MHz clock.
	if (CsHigh && frequency == EXI_CLOCK_ZERO) return false;
	// Make sure frequency is valid for this system.
	if (frequency == EXI_CLOCK_54) {
		if ((ULONG)MEM_K1_TO_PHYSICAL(ExiRegisters) == EXI_REGISTER_BASE_FLIPPER) {
			// Flipper, 54MHz frequency is not valid here.
			return false;
		}
	}

	EXI_CHANNEL_PARAMETER_REGISTER Cpr;
	Cpr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Parameter));
	// If one of the chip select lines are pulled low already, fail.
	if (Cpr.ChipSelect0 != 0 || Cpr.ChipSelect1 != 0 || Cpr.ChipSelect2 != 0) return false;
	// Could be selected for SDMC reset sequence, so check the clock frequency too.
	// If it's not zero, then some device was selected with all CS lines pulled up.
	if (Cpr.ClockFrequency != EXI_CLOCK_ZERO) return false;
	// For a memory card device (channel 1 or channel 0, device 0), if no device is present on the bus, fail.
	if ((channel == 1 || (channel == 0 && device == 0)) && !Cpr.Connected) return false;
	// Mask out all read bits other than the interrupt mask bits.
	// This has the effect of zeroing out clock frequency and all chip select bits.
	{
		EXI_CHANNEL_PARAMETER_REGISTER CprMask;
		CprMask.Value = 0;
		CprMask.DeviceInterruptMask = 1;
		CprMask.TransferInterruptMask = 1;
		CprMask.UnplugInterruptMask = 1;
		Cpr.Value &= CprMask.Value;
	}
	// Set the requested clock frequency.
	Cpr.ClockFrequency = frequency;
	// if requested, keep all CS lines pulled up (for example SDMC reset sequence needs this)
	// otherwise, pull requested CS line down
	if (!CsHigh) {
		EXI_CHANNEL_PARAMETER_REGISTER CprCs;
		CprCs.Value = 0;
		CprCs.ChipSelect0 = 1;
		Cpr.Value |= (CprCs.Value << device);
	}
	// Write the new channel parameter register value.
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Parameter), Cpr.Value);

	return true;
}

static void ExipUnselectDevice(ULONG channel, ULONG cpr) {
	// Mask out all read bits other than the interrupt mask bits.
	// This has the effect of zeroing out clock frequency and all chip select bits.
	{
		EXI_CHANNEL_PARAMETER_REGISTER CprMask;
		CprMask.Value = 0;
		CprMask.DeviceInterruptMask = 1;
		CprMask.TransferInterruptMask = 1;
		CprMask.UnplugInterruptMask = 1;
		cpr &= CprMask.Value;
	}

	// Write the new channel parameter register value.
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Parameter), cpr);
}

bool ExiUnselectDevice(ULONG channel) {
	if (channel >= EXI_CHANNEL_COUNT) return false;

	EXI_CHANNEL_PARAMETER_REGISTER Cpr;
	Cpr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Parameter));
	// If none of the chip select lines are pulled low already.
	if (Cpr.ChipSelect0 == 0 && Cpr.ChipSelect1 == 0 && Cpr.ChipSelect2 == 0) {
		// If clock frequency is zero, then there's nothing that needs to be done.
		if (Cpr.ClockFrequency == EXI_CLOCK_ZERO) return true;
	}

	ExipUnselectDevice(channel, Cpr.Value);
	return true;
}

bool ExiRefreshDevice(ULONG channel) {
	if (channel >= EXI_CHANNEL_COUNT) return false;

	// Read the channel parameter register
	EXI_CHANNEL_PARAMETER_REGISTER Cpr;
	Cpr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Parameter));
	// If none of the chip select lines are pulled low already, fail.
	if (Cpr.ChipSelect0 == 0 && Cpr.ChipSelect1 == 0 && Cpr.ChipSelect2 == 0) return false;

	// Unselect the device.
	ExipUnselectDevice(channel, Cpr.Value);

	// Write the old channel parameter register, reselecting the device.
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Parameter), Cpr.Value);
	return true;
}

static bool ExipTransferImmediate(ULONG channel, ULONG data, ULONG length, EXI_TRANSFER_TYPE type, PULONG dataRead) {
	// Shift the data into the correct bits for length.
	if (type != EXI_TRANSFER_READ) {
		data <<= (4 - length) * 8; // 4=>0, 3=>8, 2=>16, 1=>24
		MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Data), data);
	}

	// Start the transfer.
	EXI_CHANNEL_TRANSFER_REGISTER Ctr;
	Ctr.Value = 0;
	Ctr.Start = 1;
	Ctr.EnableDma = 0;
	Ctr.Type = type;
	Ctr.Length = length - 1;
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Transfer), Ctr.Value);

	// Wait for transfer to complete.
	do {
		Ctr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Transfer));
	} while (Ctr.Start);

	if (type != EXI_TRANSFER_WRITE) {
		ULONG Ret = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Data));
		Ret >>= (4 - length) * 8;
		*dataRead = Ret;
	}

	return true;
}

bool ExiTransferImmediate(ULONG channel, ULONG data, ULONG length, EXI_TRANSFER_TYPE type, PULONG dataRead) {
	if (channel >= EXI_CHANNEL_COUNT) return false;
	if (type != EXI_TRANSFER_WRITE && dataRead == NULL) return false;
	if (length > sizeof(ExiRegisters->Channel[0].Data)) return false;

	// Ensure a device is selected.
	if (!ExipDeviceSelected(channel)) return false;

	return ExipTransferImmediate(channel, data, length, type, dataRead);
}

bool ExiTransferImmediateBuffer(ULONG channel, PVOID bufferRead, PVOID bufferWrite, ULONG length, EXI_TRANSFER_TYPE type) {
	PUCHAR pRead = (PUCHAR)bufferRead;
	PUCHAR pWrite = (PUCHAR)bufferWrite;

	if (channel >= EXI_CHANNEL_COUNT) return false;
	if (type != EXI_TRANSFER_READ && pRead == NULL) return false;
	if (type != EXI_TRANSFER_WRITE && pWrite == NULL) return false;

	// Ensure a device is selected.
	if (!ExipDeviceSelected(channel)) return false;

	enum {
		EXI_BUFFER_INCREMENT = sizeof(ExiRegisters->Channel[0].Data)
	};

	ULONG thisLength = 0;
	for (; length != 0; pRead += thisLength, pWrite += thisLength, length -= thisLength) {
		thisLength = length;
		if (thisLength > EXI_BUFFER_INCREMENT)
			thisLength = EXI_BUFFER_INCREMENT;

		ULONG thisData = 0;
		if (type != EXI_TRANSFER_READ) {
			for (ULONG i = 0; i < thisLength; i++) {
				thisData |= pRead[i] << ((thisLength - i - 1) * 8);
			}
		}

		ULONG thisOutput = 0;
		if (!ExipTransferImmediate(channel, thisData, thisLength, type, &thisOutput)) return false;

		if (type != EXI_TRANSFER_WRITE) {
			for (ULONG i = 0; i < thisLength; i++) {
				pWrite[i] = (UCHAR)(thisOutput >> ((thisLength - i - 1) * 8));
			}
		}
	}

	return true;
}

bool ExiReadWriteImmediateOutBuffer(ULONG channel, UCHAR byteRead, PVOID buffer, ULONG length) {
	PUCHAR pWrite = (PUCHAR)buffer;

	if (channel >= EXI_CHANNEL_COUNT) return false;
	if (pWrite == NULL) return false;

	// Ensure a device is selected.
	if (!ExipDeviceSelected(channel)) return false;

	enum {
		EXI_BUFFER_INCREMENT = sizeof(ExiRegisters->Channel[0].Data)
	};

	ULONG wordRead = (ULONG)byteRead * 0x01010101;
	ULONG thisLength = 0;
	for (; length != 0; pWrite += thisLength, length -= thisLength) {
		thisLength = length;
		if (thisLength > EXI_BUFFER_INCREMENT)
			thisLength = EXI_BUFFER_INCREMENT;

		ULONG thisOutput = 0;
		if (!ExipTransferImmediate(channel, wordRead, thisLength, EXI_TRANSFER_READWRITE, &thisOutput)) return false;

		for (ULONG i = 0; i < thisLength; i++) {
				pWrite[i] = (UCHAR)(thisOutput >> ((thisLength - i - 1) * 8));
		}
	}

	return true;
}

// DMA transfers must be 32-byte aligned, so endian_swap64 is guaranteed to work
static void endian_swap64(void* dest, const void* src, ULONG len) {
	const ULONG* src32 = (const ULONG*)src;
	ULONG* dest32 = (ULONG*)dest;
	for (ULONG i = 0; i < len; i += sizeof(ULONG) * 2) {
		ULONG idx = i / sizeof(ULONG);
		ULONG buf0 = __builtin_bswap32(src32[idx + 0]);
		dest32[idx + 0] = __builtin_bswap32(src32[idx + 1]);
		dest32[idx + 1] = buf0;
	}
}

bool ExiTransferDma(ULONG channel, PVOID buffer, ULONG length, EXI_TRANSFER_TYPE type, EXI_SWAP_MODE swap) {
	if (channel >= EXI_CHANNEL_COUNT) return false;
	// EXI DMA does not support read+write
	if (type == EXI_TRANSFER_READWRITE) return false;
	// Check buffer / length alignment, must be 32 byte aligned
	// It's possible to do immediate transfers for the unaligned start/end of the buffer, and DMA transfer for the aligned part.
	// I won't implement that here, but for NT driver this is a possibility.
	if (((ULONG)buffer & 0x1f) != 0) return false;
	if ((length & 0x1f) != 0) return false;
	// Flipper's EXI DMA can only address the low 64MB of the address space.
	if ((ULONG)MEM_VIRTUAL_TO_PHYSICAL(buffer) >= 0x04000000 || length >= 0x04000000) {
		if ((ULONG)MEM_K1_TO_PHYSICAL(ExiRegisters) == EXI_REGISTER_BASE_FLIPPER) return false;
	}

	// Ensure a device is selected
	if (!ExipDeviceSelected(channel)) return false;

	// For a DMA write, swap the buffer if requested; then flush dcache.
	if (type == EXI_TRANSFER_WRITE) {
		if ((swap & EXI_SWAP_INPUT) != 0) {
			endian_swap64(buffer, buffer, length);
		}
		data_cache_flush(buffer, length);
	}

	// Set DMA pointer and length.
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].DmaAddress), (ULONG)MEM_VIRTUAL_TO_PHYSICAL(buffer));
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].DmaLength), length);

	// Start the transfer.
	EXI_CHANNEL_TRANSFER_REGISTER Ctr;
	Ctr.Value = 0;
	Ctr.Start = 1;
	Ctr.EnableDma = 1;
	Ctr.Type = type;
	Ctr.Length = 0;
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Transfer), Ctr.Value);

	// Wait for transfer to complete.
	do {
		Ctr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Transfer));
	} while (Ctr.Start);

	// If the buffer got written to by DMA, invalidate data cache.
	if (type == EXI_TRANSFER_READ) {
		data_cache_invalidate(buffer, length);
	}

	// If requested, swap the buffer.
	if ((type == EXI_TRANSFER_READ && swap != EXI_SWAP_NONE) || (type == EXI_TRANSFER_WRITE && (swap & EXI_SWAP_OUTPUT) != 0)) {
		endian_swap64(buffer, buffer, length);
	}

	// All done.
	return true;
}

bool ExiReadDmaWithImmediate(ULONG channel, UCHAR immediate, PVOID buffer, ULONG length, EXI_SWAP_MODE swap) {
	if (channel >= EXI_CHANNEL_COUNT) return false;
	// Check buffer / length alignment, must be 32 byte aligned
	if (((ULONG)buffer & 0x1f) != 0) return false;
	if ((length & 0x1f) != 0) return false;
	// Flipper's EXI DMA can only address the low 64MB of the address space.
	if ((ULONG)MEM_VIRTUAL_TO_PHYSICAL(buffer) >= 0x04000000 || length >= 0x04000000) {
		if ((ULONG)MEM_K1_TO_PHYSICAL(ExiRegisters) == EXI_REGISTER_BASE_FLIPPER) return false;
	}

	// Ensure a device is selected
	if (!ExipDeviceSelected(channel)) return false;

	// Set the data register to immediate value. This doesn't work under emulation but should under real hardware, at least for the first 32 bits transferred.
	// Swiss uses this functionality for DMA-reads of SD cards in SPI mode, so I assume it does work as intended:
	ULONG imm32 = (ULONG)immediate * 0x01010101;
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Data), imm32);

	// Set DMA pointer and length.
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].DmaAddress), (ULONG)MEM_VIRTUAL_TO_PHYSICAL(buffer));
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].DmaLength), length);

	// Start the transfer.
	EXI_CHANNEL_TRANSFER_REGISTER Ctr;
	Ctr.Value = 0;
	Ctr.Start = 1;
	Ctr.EnableDma = 1;
	Ctr.Type = EXI_TRANSFER_READ;
	Ctr.Length = 0;
	MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Transfer), Ctr.Value);

	// Wait for transfer to complete.
	do {
		Ctr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Transfer));
	} while (Ctr.Start);

	// The buffer got overwritten by DMA, invalidate data cache.
	data_cache_invalidate(buffer, length);

	// If requested, swap the buffer.
	if (swap != EXI_SWAP_NONE) {
		endian_swap64(buffer, buffer, length);
	}

	// All done.
	return true;
}

ULONG ExiReadDataRegister(ULONG channel) {
	if (channel >= EXI_CHANNEL_COUNT) return 0;
	return MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[channel].Data));
}

bool ExiBufferCanDma(PVOID buffer, ULONG length) {
	if (((ULONG)buffer & 0x1f) != 0) return false;
	if ((length & 0x1f) != 0) return false;

	return true;
}

static bool ExipGetDeviceIdentifierAttempt(ULONG channel, ULONG device, PULONG deviceIdentifier) {
	enum {
		EXI_CMD_DEVICE_ID = 0
	};

	if (!ExiSelectDevice(channel, device, EXI_CLOCK_0_8, false)) return false;
	if (!ExipTransferImmediate(channel, EXI_CMD_DEVICE_ID, 2, EXI_TRANSFER_WRITE, NULL)) return false;
	if (!ExipTransferImmediate(channel, 0, sizeof(*deviceIdentifier), EXI_TRANSFER_READ, deviceIdentifier)) return false;
	if (!ExiUnselectDevice(channel)) return false;
	return true;
}

bool ExiGetDeviceIdentifier(ULONG channel, ULONG device, PULONG deviceIdentifier) {
	if (deviceIdentifier == NULL) return false;
	if (channel >= EXI_CHANNEL_COUNT) return false;

	// If this device is a mounted SD card, don't touch it.
	EXI_SDMC_DRIVE drive;
	if (channel == 0) {
		if (device == 0) drive = SDMC_DRIVE_CARD_A;
		if (device == 2) drive = SDMC_DRIVE_SP1;
	}
	else {
		if (channel == 1) drive = SDMC_DRIVE_CARD_B;
		if (channel == 2) drive = SDMC_DRIVE_SP2;
	}
	if (SdmcexiIsMounted(drive)) {
		*deviceIdentifier = 0xFFFFFFFF;
		return true;
	}

	ULONG thisId = 0xFFFFFFFF;
	// Try several times and make sure the same identifier is obtained twice in a row.
	// Give up after many tries and return the last obtained identifier.
	for (ULONG attempt = 0; attempt < 4; attempt++) {
		if (!ExipGetDeviceIdentifierAttempt(channel, device, deviceIdentifier)) return false;
		if (thisId == *deviceIdentifier) break;
		thisId = *deviceIdentifier;
	}

	return true;
}

void ExiInit(ARTX_SYSTEM_TYPE SystemType) {
#if 0
	// Assert that all bitfield structures are correct.
	// This optimises down to a constant, so:
	LARGE_INTEGER BitmaskTest;
	BitmaskTest.QuadPart = ExipBitmaskTest();
	if (BitmaskTest.HighPart >= 0) {
		printf("Bitmask test %x failed - failing value %08x\r\n", BitmaskTest.HighPart, BitmaskTest.LowPart);
		while (1);
	}
#endif

	// Initialise EXI register address, map the correct physaddr for this system
	ExiRegisters = (PEXI_REGISTERS)MEM_PHYSICAL_TO_K1(ExipGetPhysBase(SystemType));

	// Wait until no transfer is in progress on all channels
	_Static_assert(EXI_CHANNEL_COUNT == 3);
	{
		EXI_CHANNEL_TRANSFER_REGISTER Tcr;
		// Unroll this loop manually, so compiler can do contant-folding on the register offset
		
		// channel 0
		do {
			Tcr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[0].Transfer));
		} while (Tcr.Start == 1);
		// channel 1
		do {
			Tcr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[1].Transfer));
		} while (Tcr.Start == 1);
		// channel 2
		do {
			Tcr.Value = MmioReadBase32(MMIO_OFFSET(ExiRegisters, Channel[2].Transfer));
		} while (Tcr.Start == 1);
	}

	// Initialise all channels to zero out all writable bits.
	{
		EXI_CHANNEL_PARAMETER_REGISTER Cpr;
		Cpr.Value = 0;
		// For channel 0, ensure the bootrom disable bit is set, to ensure flipper's LFSR+XOR cryptoscheme doesn't get in the way.
		Cpr.BootromDisable = 1;
		MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[0].Parameter), Cpr.Value);
		// That bit only exists on channel 0, so unset it for the other channels.
		Cpr.Value = 0;
		MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[1].Parameter), Cpr.Value);
		MmioWriteBase32(MMIO_OFFSET(ExiRegisters, Channel[2].Parameter), Cpr.Value);
	}


}

static int64_t ExipBitmaskTest(void) {
#define REGISTER_TEST(Var, Expected) \
	Ret.LowPart = Var .Value; \
	if (Var .Value != (ULONG)(Expected)) return Ret.QuadPart; \
	Ret.HighPart++; \
	Var .Value = 0;

	LARGE_INTEGER Ret;
	Ret.HighPart = 0;

	// Parameter Register
	{
		EXI_CHANNEL_PARAMETER_REGISTER Cpr;
		Cpr.Value = 0;

		Cpr.DeviceInterruptMask = 1;
		REGISTER_TEST(Cpr, ARC_BIT(0)); // 0
		Cpr.DeviceInterruptStatus = 1;
		REGISTER_TEST(Cpr, ARC_BIT(1)); // 1
		Cpr.TransferInterruptMask = 1;
		REGISTER_TEST(Cpr, ARC_BIT(2)); // 2
		Cpr.TransferInterruptStatus = 1;
		REGISTER_TEST(Cpr, ARC_BIT(3)); // 3
		Cpr.ClockFrequency = (EXI_CLOCK_FREQUENCY)7;
		REGISTER_TEST(Cpr, 0x00000070); // 4
		Cpr.ChipSelect0 = 1;
		REGISTER_TEST(Cpr, ARC_BIT(7)); // 5
		Cpr.ChipSelect1 = 1;
		REGISTER_TEST(Cpr, ARC_BIT(8)); // 6
		Cpr.ChipSelect2 = 1;
		REGISTER_TEST(Cpr, ARC_BIT(9)); // 7
		Cpr.UnplugInterruptMask = 1;
		REGISTER_TEST(Cpr, ARC_BIT(10)); // 8
		Cpr.UnplugInterruptStatus = 1;
		REGISTER_TEST(Cpr, ARC_BIT(11)); // 9
		Cpr.Connected = 1;
		REGISTER_TEST(Cpr, ARC_BIT(12)); // a
		Cpr.BootromDisable = 1;
		REGISTER_TEST(Cpr, ARC_BIT(13)); // b
	}

	// Transfer Control Register
	{
		EXI_CHANNEL_TRANSFER_REGISTER Tcr;
		Tcr.Value = 0;

		Tcr.Start = 1;
		REGISTER_TEST(Tcr, ARC_BIT(0)); // c
		Tcr.EnableDma = 1;
		REGISTER_TEST(Tcr, ARC_BIT(1)); // d
		Tcr.Type = (EXI_TRANSFER_TYPE)3;
		REGISTER_TEST(Tcr, 0x0000000C); // e
		Tcr.Length = 3;
		REGISTER_TEST(Tcr, 0x00000030); // e
	}

	Ret.HighPart = Ret.LowPart = 0xFFFFFFFF;
	return Ret.QuadPart;

#undef REGISTER_TEST
}
