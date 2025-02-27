// Dolphin Serial Interface. Used to handle gamecube controller (and ASCII keyboard controller) support.
// Base address is at 0x0C006400 on flipper, 0x0D006400 on vegas and latte.
// Dolphin (emulator) sets up both areas when emulating broadway + parts of vegas, but vegas real hw only allows for one or the other mapping to be used (and with IOS drivers running, we need the second)

#include <stdio.h>
#include <memory.h>
#include "arc.h"
#include "types.h"
#include "runtime.h"
#include "si.h"

// SI register definitions.

// Please note: any arrays in these unions will use little endian ordering due to MSR_LE swizzle or optimisation.
// So array[4] => correct big endian index is array[3 - index]

typedef union _SI_CHANNEL_COMMAND_REGISTER {
	struct {
		UCHAR Data2, Data1, Data0, Pad;
	};
	ULONG Value;
} SI_CHANNEL_COMMAND_REGISTER;

typedef union _SI_CHANNEL_RESPONSE_HIGH_REGISTER {
	struct {
		UCHAR Data3, Data2, Data1;

		UCHAR Data0 : 6;
		BOOLEAN ErrorLatch : 1;
		BOOLEAN ErrorStatus : 1;
	};

	ULONG Value;
} SI_CHANNEL_RESPONSE_HIGH_REGISTER;

typedef union _SI_CHANNEL_RESPONSE_LOW_REGISTER {
	struct {
		UCHAR Data3, Data2, Data1, Data0;
	};
	ULONG Value;
} SI_CHANNEL_RESPONSE_LOW_REGISTER;

typedef struct ARC_ALIGNED(4) _SI_CHANNEL_REGISTERS {
	SI_CHANNEL_COMMAND_REGISTER Command;
	SI_CHANNEL_RESPONSE_HIGH_REGISTER ResponseHigh;
	SI_CHANNEL_RESPONSE_LOW_REGISTER ResponseLow;
} SI_CHANNEL_REGISTERS, *PSI_CHANNEL_REGISTERS;

typedef union _SI_POLL_REGISTER {
	struct {
		BOOLEAN
			VblankCopy3 : 1,
			VblankCopy2 : 1,
			VblankCopy1 : 1,
			VblankCopy0 : 1;
		BOOLEAN
			Enable3 : 1,
			Enable2 : 1,
			Enable1 : 1,
			Enable0 : 1;

		UCHAR YTimes;

		USHORT XLines : 10;
		USHORT : 6;
	};
	ULONG Value;
} SI_POLL_REGISTER, *PSI_POLL_REGISTER;

typedef union _SI_COMMUNICATION_CONTROL_STATUS_REGISTER {
	struct {
		BOOLEAN TransferStart : 1;
		UCHAR Channel : 2;
		UCHAR : 5;

		UCHAR InputLength : 7;
		UCHAR : 1;

		UCHAR OutputLength : 7;
		UCHAR : 1;

		UCHAR : 3;
		BOOLEAN ReadStatusInterruptMask : 1;
		BOOLEAN ReadStatusInterruptStatus : 1;
		BOOLEAN CommunicationError : 1;
		BOOLEAN TransferCompleteInterruptMask : 1;
		BOOLEAN TransferCompleteInterruptStatus : 1;
	};
	ULONG Value;
} SI_COMMUNICATION_CONTROL_STATUS_REGISTER, *PSI_COMMUNICATION_CONTROL_STATUS_REGISTER;

typedef union _SI_STATUS_REGISTER_CHANNEL {
	struct {
		BOOLEAN ErrorUnderrun : 1;
		BOOLEAN ErrorOverrun : 1;
		BOOLEAN ErrorCollision : 1;
		BOOLEAN ErrorNoResponse : 1;
		BOOLEAN WriteStatus : 1;
		BOOLEAN ReadStatus : 1;
		UCHAR : 2;
	};
	UCHAR Value;
} SI_STATUS_REGISTER_CHANNEL, *PSI_STATUS_REGISTER_CHANNEL;

typedef union _SI_STATUS_REGISTER {
	struct {
		ULONG : 31;
		BOOLEAN WriteAll : 1;
	};
	struct {
		SI_STATUS_REGISTER_CHANNEL Channel3, Channel2, Channel1, Channel0;
	};
	struct {
		SI_STATUS_REGISTER_CHANNEL Channels[4];
	};
	ULONG Value;
} SI_STATUS_REGISTER, *PSI_STATUS_REGISTER;

typedef union _SI_EXI_CLOCK_LOCK_REGISTER {
	struct {
		ULONG : 31;
		BOOLEAN Enabled : 1;
	};
	ULONG Value;
} SI_EXI_CLOCK_LOCK_REGISTER, *PSI_EXI_CLOCK_LOCK_REGISTER;

typedef struct _SI_REGISTERS {
	// 0x00
	SI_CHANNEL_REGISTERS Channel[SI_CHANNEL_COUNT];
	// 0x30
	SI_POLL_REGISTER Poll;
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER CommunicationControlStatus;
	SI_STATUS_REGISTER Status;
	SI_EXI_CLOCK_LOCK_REGISTER ClockLock;
	// 0x40
	ULONG Reserved[0x40 / sizeof(ULONG)];
	// 0x80
	ULONG IoBuffer[0x80 / sizeof(ULONG)];
	// 0x100
} SI_REGISTERS, *PSI_REGISTERS;

_Static_assert(sizeof(SI_REGISTERS) == 0x100);

enum {
	SI_MAX_TRANSFER_LENGTH = sizeof(((PSI_REGISTERS)NULL)->IoBuffer)
};

enum {
	SI_REGISTER_BASE_FLIPPER = 0x0C006400,
	SI_REGISTER_OFFSET_VEGAS = (0x0D006400 - SI_REGISTER_BASE_FLIPPER)
};

static __attribute__((const)) inline ULONG SipGetPhysBase(ARTX_SYSTEM_TYPE SystemType)  {
	ULONG base = SI_REGISTER_BASE_FLIPPER;
	if (SystemType != ARTX_SYSTEM_FLIPPER) base += SI_REGISTER_OFFSET_VEGAS;
	return base;
}

static PSI_REGISTERS SiRegisters;

static ULONG part32_pack_big(PVOID buffer, ULONG length) {
	PUCHAR buffer8 = (PUCHAR)buffer;

	ULONG ret = 0;
	for (ULONG i = 0; i < length; i++) {
		ret |= ((ULONG)buffer8[i]) << ((3 - i) * 8);
	}

	return ret;
}

static void part32_unpack_big(ULONG value, PVOID buffer, ULONG length) {
	PUCHAR buffer8 = (PUCHAR)buffer;

	for (ULONG i = 0; i < length; i++) {
		buffer8[3 - i] = (UCHAR)(value >> ((3 - i) * 8));
	}
}

static void SipBufferCopy(PVOID buffer, ULONG length, bool writeIo) {
	if (length == 0) length = SI_MAX_TRANSFER_LENGTH;
	ULONG alignedLen = length & (sizeof(SiRegisters->IoBuffer[0]) - 1);
	ULONG additional = alignedLen;
	alignedLen = length + sizeof(SiRegisters->IoBuffer[0]) - alignedLen;
	ULONG count = alignedLen / sizeof(SiRegisters->IoBuffer[0]);
	if (additional != 0) count--;

	PULONG buffer32 = (PULONG)buffer;

	if (writeIo) {
		for (ULONG i = 0; i < count; i++) {
			MmioWriteBase32(&SiRegisters->IoBuffer, __builtin_offsetof(SI_REGISTERS, IoBuffer[i]) - __builtin_offsetof(SI_REGISTERS, IoBuffer), buffer32[i]);
		}
		if (additional == 0) return;
		ULONG value = part32_pack_big(&buffer32[count], additional);
		MmioWriteBase32(&SiRegisters->IoBuffer, __builtin_offsetof(SI_REGISTERS, IoBuffer[count]) - __builtin_offsetof(SI_REGISTERS, IoBuffer), value);
	}
	else {
		for (ULONG i = 0; i < count; i++) {
			buffer32[i] = MmioReadBase32(&SiRegisters->IoBuffer, __builtin_offsetof(SI_REGISTERS, IoBuffer[i]) - __builtin_offsetof(SI_REGISTERS, IoBuffer));
		}
		if (additional == 0) return;
		ULONG value = MmioReadBase32(&SiRegisters->IoBuffer, __builtin_offsetof(SI_REGISTERS, IoBuffer[count]) - __builtin_offsetof(SI_REGISTERS, IoBuffer));
		part32_unpack_big(value, &buffer32[count], additional);
	}
}


bool SiTransferSync(ULONG channel, PVOID bufWrite, ULONG lenWrite, PVOID bufRead, ULONG lenRead) {
	if (channel >= SI_CHANNEL_COUNT) return false;
	if (lenWrite > SI_MAX_TRANSFER_LENGTH || lenRead > SI_MAX_TRANSFER_LENGTH) return false;
	if (lenWrite == 0 || lenRead == 0) return false;

	// Invert the channel number to get the correct index into the array.
	// Could use MmioWrite8/etc, but this is one instruction.
	ULONG chanInvert = (SI_CHANNEL_COUNT - 1) - channel;

	// Clear errors for the specified channel
	SI_STATUS_REGISTER StatusMask;
	StatusMask.Value = 0;
	StatusMask.Channels[chanInvert].ErrorNoResponse = 1;
	StatusMask.Channels[chanInvert].ErrorCollision = 1;
	StatusMask.Channels[chanInvert].ErrorOverrun = 1;
	StatusMask.Channels[chanInvert].ErrorUnderrun = 1;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, Status),
		MmioReadBase32(MMIO_OFFSET(SiRegisters, Status)) & StatusMask.Value
	);

	// Copy the data to write to the SI I/O buffer
	SipBufferCopy(bufWrite, lenWrite, true);

	if (lenWrite == SI_MAX_TRANSFER_LENGTH) lenWrite = 0;
	if (lenRead == SI_MAX_TRANSFER_LENGTH) lenRead = 0;

	// Configure and start the transfer
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER Comcs;
	Comcs.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus));
	Comcs.TransferCompleteInterruptStatus = 1; // Clear interrupt
	Comcs.TransferCompleteInterruptMask = 0; // Mask out interrupt (for synchronous transfer)
	Comcs.OutputLength = lenWrite;
	Comcs.InputLength = lenRead;
	Comcs.Channel = channel;
	Comcs.TransferStart = 1;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus), Comcs.Value);

	// Wait for transfer...
	do {
		Comcs.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus));
	} while (Comcs.TransferStart);

	// Read status register
	SI_STATUS_REGISTER Status;
	Status.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, Status));

	// Acknowledge interrupt
	Comcs.TransferCompleteInterruptStatus = 1;
	Comcs.TransferStart = 0;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus), Comcs.Value);

	// Copy the data from the SI I/O buffer to the output buffer
	SipBufferCopy(bufRead, lenRead, false);

	return (Comcs.CommunicationError == 0);
}

static JOYBUS_DEVICE_TYPE SipGetDeviceType(ULONG channel, UCHAR command) {
	JOYBUS_DEVICE_TYPE Type;
	Type.Value = 0;
	if (!SiTransferByteSync(channel, command, &Type.Value, sizeof(Type) - 1)) {
		Type.Value = 0xFFFFFFFF;
	}
	return Type;
}

JOYBUS_DEVICE_TYPE SiGetDeviceType(ULONG channel) {
	return SipGetDeviceType(channel, 0x00);
}

JOYBUS_DEVICE_TYPE SiGetDeviceTypeReset(ULONG channel) {
	return SipGetDeviceType(channel, 0xFF);
}

static int64_t SipBitmaskTest(void);

void SiInit(ARTX_SYSTEM_TYPE SystemType) {
#if 0
	// Assert that all bitfield structures are correct.
	// This optimises down to a constant, so:
	LARGE_INTEGER BitmaskTest;
	BitmaskTest.QuadPart = SipBitmaskTest();
	if (BitmaskTest.HighPart >= 0) {
		printf("Bitmask test %x failed - failing value %08x\r\n", BitmaskTest.HighPart, BitmaskTest.LowPart);
		while (1);
	}
#endif

	SiRegisters = (PSI_REGISTERS)MEM_PHYSICAL_TO_K1(SipGetPhysBase(SystemType));
	// TODO: set default sampling rate, different value for 640x480 or 640x528
	
	// Wait for any transfer to complete
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER Comcs;
	do {
		Comcs.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus));
	} while (Comcs.TransferStart);

	// Disable interrupts and acknowledge transfer complete interrupt
	Comcs.Value = 0;
	Comcs.TransferCompleteInterruptStatus = 1;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus), Comcs.Value);

}

static int64_t SipBitmaskTest(void) {
#define REGISTER_TEST(Var, Expected) \
	Ret.LowPart = Var .Value; \
	if (Var .Value != (ULONG)(Expected)) return Ret.QuadPart; \
	Ret.HighPart++; \
	Var .Value = 0;

	LARGE_INTEGER Ret;
	Ret.HighPart = 0;

	// ChannelCommand
	{
		SI_CHANNEL_COMMAND_REGISTER Cmd;
		Cmd.Value = 0;
		Cmd.Pad = 0xFF;
		REGISTER_TEST(Cmd, 0xFF000000ul); // 0
		Cmd.Data0 = 0xFF;
		REGISTER_TEST(Cmd, 0x00FF0000ul); // 1
		Cmd.Data1 = 0xFF;
		REGISTER_TEST(Cmd, 0x0000FF00ul); // 2
		Cmd.Data2 = 0xFF;
		REGISTER_TEST(Cmd, 0x000000FFul); // 3
	}

	// ChannelResponseHigh
	{
		SI_CHANNEL_RESPONSE_HIGH_REGISTER Crh;
		Crh.Value = 0;
		Crh.ErrorStatus = 1;
		REGISTER_TEST(Crh, ARC_BIT(31)); // 4
		Crh.ErrorLatch = 1;
		REGISTER_TEST(Crh, ARC_BIT(30)); // 5
		Crh.Data0 = 0x3F;
		REGISTER_TEST(Crh, 0x3F000000ul); // 6
		Crh.Data1 = 0xFF;
		REGISTER_TEST(Crh, 0x00FF0000ul); // 7
		Crh.Data2 = 0xFF;
		REGISTER_TEST(Crh, 0x0000FF00ul); // 8
		Crh.Data3 = 0xFF;
		REGISTER_TEST(Crh, 0x000000FFul); // 9
	}

	// ChannelResponseLow
	{
		SI_CHANNEL_RESPONSE_LOW_REGISTER Crl;
		Crl.Value = 0;
		Crl.Data0 = 0xFF;
		REGISTER_TEST(Crl, 0xFF000000ul); // a
		Crl.Data1 = 0xFF;
		REGISTER_TEST(Crl, 0x00FF0000ul); // b
		Crl.Data2 = 0xFF;
		REGISTER_TEST(Crl, 0x0000FF00ul); // c
		Crl.Data3 = 0xFF;
		REGISTER_TEST(Crl, 0x000000FFul); // d
	}

	// Poll
	{
		SI_POLL_REGISTER Poll;
		Poll.Value = 0;
		Poll.VblankCopy0 = 1;
		REGISTER_TEST(Poll, ARC_BIT(3)); // e
		Poll.VblankCopy1 = 1;
		REGISTER_TEST(Poll, ARC_BIT(2)); // f
		Poll.VblankCopy2 = 1;
		REGISTER_TEST(Poll, ARC_BIT(1)); // 10
		Poll.VblankCopy3 = 1;
		REGISTER_TEST(Poll, ARC_BIT(0)); // 11
		Poll.Enable0 = 1;
		REGISTER_TEST(Poll, ARC_BIT(7)); // 12
		Poll.Enable1 = 1;
		REGISTER_TEST(Poll, ARC_BIT(6)); // 13
		Poll.Enable2 = 1;
		REGISTER_TEST(Poll, ARC_BIT(5)); // 14
		Poll.Enable3 = 1;
		REGISTER_TEST(Poll, ARC_BIT(4)); // 15
		Poll.YTimes = 0xFF;
		REGISTER_TEST(Poll, 0x0000FF00ul); // 16
		Poll.XLines = 0x3FF;
		REGISTER_TEST(Poll, 0x03FF0000ul); // 17
	}

	// Csr
	{
		SI_COMMUNICATION_CONTROL_STATUS_REGISTER Csr;
		Csr.Value = 0;
		Csr.TransferStart = 1;
		REGISTER_TEST(Csr, ARC_BIT(0)); // 18
		Csr.Channel = 3;
		REGISTER_TEST(Csr, 0x00000006ul); // 19
		Csr.InputLength = 0x7F;
		REGISTER_TEST(Csr, 0x00007f00ul); // 1a
		Csr.OutputLength = 0x7F;
		REGISTER_TEST(Csr, 0x007f0000ul); // 1b
		Csr.ReadStatusInterruptMask = 1;
		REGISTER_TEST(Csr, ARC_BIT(27)); // 1c
		Csr.ReadStatusInterruptStatus = 1;
		REGISTER_TEST(Csr, ARC_BIT(28)); // 1d
		Csr.CommunicationError = 1;
		REGISTER_TEST(Csr, ARC_BIT(29)); // 1e
		Csr.TransferCompleteInterruptMask = 1;
		REGISTER_TEST(Csr, ARC_BIT(30)); // 1f
		Csr.TransferCompleteInterruptStatus = 1;
		REGISTER_TEST(Csr, ARC_BIT(31)); // 20
	}

	// StatusChannel
	{
		SI_STATUS_REGISTER_CHANNEL Channel;
		Channel.Value = 0;
		Channel.ErrorUnderrun = 1;
		REGISTER_TEST(Channel, ARC_BIT(0)); // 21
		Channel.ErrorOverrun = 1;
		REGISTER_TEST(Channel, ARC_BIT(1)); // 22
		Channel.ErrorCollision = 1;
		REGISTER_TEST(Channel, ARC_BIT(2)); // 23
		Channel.ErrorNoResponse = 1;
		REGISTER_TEST(Channel, ARC_BIT(3)); // 24
		Channel.WriteStatus = 1;
		REGISTER_TEST(Channel, ARC_BIT(4)); // 25
		Channel.ReadStatus = 1;
		REGISTER_TEST(Channel, ARC_BIT(5)); // 26
	}

	// Status
	{
		SI_STATUS_REGISTER Status;
		Status.Value = 0;
		Status.WriteAll = 1;
		REGISTER_TEST(Status, ARC_BIT(31)); // 27
		PSI_STATUS_REGISTER_CHANNEL Channel = (PSI_STATUS_REGISTER_CHANNEL)&Status;
		Channel[3 - 0].ErrorUnderrun = 1;
		REGISTER_TEST(Status, ARC_BIT(24)); // 28
		Channel[3 - 1].ErrorUnderrun = 1;
		REGISTER_TEST(Status, ARC_BIT(16)); // 29
		Channel[3 - 2].ErrorUnderrun = 1;
		REGISTER_TEST(Status, ARC_BIT(8)); // 30
		Channel[3 - 3].ErrorUnderrun = 1;
		REGISTER_TEST(Status, ARC_BIT(0)); // 31

	}

	Ret.HighPart = Ret.LowPart = 0xFFFFFFFF;
	return Ret.QuadPart;

#undef REGISTER_TEST
}