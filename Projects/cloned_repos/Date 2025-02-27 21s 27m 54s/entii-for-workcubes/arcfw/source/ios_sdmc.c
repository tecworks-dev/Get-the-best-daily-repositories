// RVL/Cafe SD slot by IOS.
#include <stdio.h>
#include <memory.h>
#include "arc.h"
#include "types.h"
#include "runtime.h"
#include "pxi.h"
#include "timer.h"

// Type definitions for IOS SDMC interface.
typedef enum {
	SDIOHCR_RESPONSE = 0x10,
	SDIOHCR_HOSTCONTROL = 0x28,
	SDIOHCR_POWERCONTROL = 0x29,
	SDIOHCR_CLOCKCONTROL = 0x2C,
	SDIOHCR_TIMEOUTCONTROL = 0x2E,
	SDIOHCR_SOFTWARERESET = 0x2F,

	SDIOHCR_HOSTCONTROL_4BIT = 0x02
} SDIO_CONTROLLER_REGISTER;

typedef enum {
	SDIO_CMD_GOIDLE = 0,
	SDIO_CMD_ALL_SENDCID = 2,
	SDIO_CMD_SENDRCA = 3,
	SDIO_CMD_SELECT = 7,
	SDIO_CMD_DESELECT = 7,
	SDIO_CMD_SENDIFCOND = 8,
	SDIO_CMD_SENDCSD = 9,
	SDIO_CMD_SENDCID = 10,
	SDIO_CMD_SENDSTATUS = 13,
	SDIO_CMD_SETBLOCKLEN = 16,
	SDIO_CMD_READBLOCK = 17,
	SDIO_CMD_READMULTIBLOCK = 18,
	SDIO_CMD_WRITEBLOCK = 24,
	SDIO_CMD_WRITEMULTIBLOCK = 25,
	SDIO_CMD_APPCMD = 0x37,

	SDIO_ACMD_SETBUSWIDTH = 0x06,
	SDIO_ACMD_SENDSCR = 0x33,
	SDIO_ACMD_SENDOPCOND = 0x29
} SDIO_COMMAND;

typedef enum {
	SDIO_TYPE_NONE,
	SDIO_TYPE_READ,
	SDIO_TYPE_WRITE,
	SDIO_TYPE_CMD
} SDIO_COMMAND_TYPE;

typedef enum {
	SDIO_RESPONSE_NONE,
	SDIO_RESPONSE_R1,
	SDIO_RESPONSE_R1B,
	SDIO_RESPONSE_R2,
	SDIO_RESPONSE_R3,
	SDIO_RESPONSE_R4,
	SDIO_RESPONSE_R5,
	SDIO_RESPONSE_R6,
} SDIO_RESPONSE;

typedef enum {
	SDIO_STATUS_CARD_INSERTED = BIT(0),
	SDIO_STATUS_CARD_REMOVED = BIT(1),
	SDIO_STATUS_CARD_WRITEPROT = BIT(2),
	SDIO_STATUS_SUSPEND = BIT(3),
	SDIO_STATUS_BUSY = BIT(4),

	SDIO_TYPE_UNKNOWN = 0,
	SDIO_TYPE_MEMORY = (BIT(0) << 16),
	SDIO_TYPE_SDIO = (BIT(1) << 16),
	SDIO_TYPE_COMBO = (BIT(2) << 16),
	SDIO_TYPE_MMC = (BIT(3) << 16),
	SDIO_TYPE_SDHC = (BIT(4) << 16),
} SDIO_STATUS;

enum {
	SDIO_DEFAULT_TIMEOUT = 0xe
};

enum {
	SDMC_BUFFER_PHYS_START = 0x10000000,
	SDMC_BUFFER_LENGTH = 0x80000,
	SDMC_BUFFER_PHYS_END = SDMC_BUFFER_PHYS_START + SDMC_BUFFER_LENGTH
};

enum {
	SDMC_SECTOR_SIZE = 0x200,
	// Single tasking bare metal firmware, so use the entire length of the buffer.
	SDMC_SECTORS_IN_PAGE = SDMC_BUFFER_LENGTH / SDMC_SECTOR_SIZE
};

typedef enum {
	IOCTL_SDIO_WRITEHCREG = 1,
	IOCTL_SDIO_READHCREG,
	IOCTL_SDIO_READCREG,
	IOCTL_SDIO_RESETCARD,
	IOCTL_SDIO_WRITECREF,
	IOCTL_SDIO_SETCLK,
	IOCTL_SDIO_SENDCMD,
	IOCTL_SDIO_SETBUSWIDTH,
	IOCTL_SDIO_READMCREG,
	IOCTL_SDIO_WRITEMCREG,
	IOCTL_SDIO_GETSTATUS,
	IOCTL_SDIO_GETOCR,
	IOCTL_SDIO_READDATA,
	IOCTL_SDIO_WRITEDATA
} IOS_IOCTL_SDMC;

typedef struct _IOS_SDMC_COMMAND {
	ULONG CommandType;
	ULONG Command;

	ULONG Argument;
	ULONG ResponseType;

	ULONG BlockSize;
	ULONG BlockCount;

	ULONG IsDma;
	ULONG UserBuffer;

	ULONG Padding1;
	ULONG Padding0;
} IOS_SDMC_COMMAND, *PIOS_SDMC_COMMAND;

typedef struct _IOS_SDMC_MMIO {
	ULONG BlockSize;
	ULONG Address;
	ULONG Width;
	ULONG BlockCount;
	ULONG IsDma;
	ULONG Value;
} IOS_SDMC_MMIO, *PIOS_SDMC_MMIO;

typedef struct _IOS_SDMC_RESPONSE {
	ULONG Field1;
	ULONG Field0;
	LONG ACmd12Response;
	ULONG Field2;
} IOS_SDMC_RESPONSE, *PIOS_SDMC_RESPONSE;

// Command buffer, everything correctly aligned for IOS IPC
typedef struct _IOS_SDMC_SEND_COMMAND_BUFFER {
	IOS_IOCTL_VECTOR Vectors[3] ARC_ALIGNED(32);
	IOS_SDMC_COMMAND Command ARC_ALIGNED(32);
	IOS_SDMC_RESPONSE Response ARC_ALIGNED(32);
} IOS_SDMC_SEND_COMMAND_BUFFER, * PIOS_SDMC_SEND_COMMAND_BUFFER;

static IOS_HANDLE s_hIosSdmc = IOS_HANDLE_INVALID;
#define s_SdmcDmaBuffer MEM_PHYSICAL_TO_K1(SDMC_BUFFER_PHYS_START)

static bool s_SdmcIsHighCapacity = false;
static bool s_SdmcIsInitialised = false;

static USHORT s_SdmcRca = 0;

static UCHAR s_SdmcCid[16] ARC_ALIGNED(8);
static UCHAR s_SdmcCsd[16] ARC_ALIGNED(8);

static void SdmcpCopyResponse(PVOID Destination, PIOS_SDMC_RESPONSE Source, ULONG Length) {
	if ((Length & 3) != 0) return;
	if (Length > sizeof(IOS_SDMC_RESPONSE)) Length = sizeof(IOS_SDMC_RESPONSE) / sizeof(ULONG);
	else Length /= sizeof(ULONG);
	PULONG Buffer32 = (PULONG)Destination;

	ULONG Index = 0;
	if (Length == 0) return;
	Buffer32[Index++] = Source->Field0;
	if ((--Length) == 0) return;
	Buffer32[Index++] = Source->Field1;
	if ((--Length) == 0) return;
	Buffer32[Index++] = Source->Field2;
	if ((--Length) == 0) return;
	Buffer32[Index++] = Source->ACmd12Response;
}

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

static void endian_swap64by8(void* dest, const void* src, ULONG len) {
	const UCHAR* src8 = (const UCHAR*)src;
	UCHAR* dest8 = (UCHAR*)dest;

	UCHAR arr[8];
	for (ULONG i = 0; i < len; i += sizeof(ULONG) * 2) {
		memcpy(arr, &src8[i], sizeof(arr));

		NativeWrite8(&dest8[0], arr[0]);
		NativeWrite8(&dest8[1], arr[1]);
		NativeWrite8(&dest8[2], arr[2]);
		NativeWrite8(&dest8[3], arr[3]);
		NativeWrite8(&dest8[4], arr[4]);
		NativeWrite8(&dest8[5], arr[5]);
		NativeWrite8(&dest8[6], arr[6]);
		NativeWrite8(&dest8[7], arr[7]);
	}
}

static void memcpy32(void* dest, const void* src, ULONG len) {
	PULONG dest32 = (PULONG)dest;
	const ULONG* src32 = (const ULONG*)src;

	if ((len & 3) != 0) return;

	len /= sizeof(ULONG);
	for (ULONG i = 0; i < len; i++) dest32[i] = src32[i];
}

static void SdmcpToggleLed(void) {
	MmioWriteBase32(MEM_PHYSICAL_TO_K1(0x0d800000), 0xc0, MmioReadBase32(MEM_PHYSICAL_TO_K1(0x0d800000), 0xc0) ^ 0x20);
}

static LONG SdmcpSendCommandImpl(PIOS_SDMC_SEND_COMMAND_BUFFER CmdBuf, BOOLEAN CanTimeoutOnIosSide) {
	(void)CanTimeoutOnIosSide;

	// Assumption: don't need to deal with endianness swap here, it will be dealt with further up the call stack for reads/writes.
	if (CmdBuf->Vectors[0].Pointer != NULL) {
		LONG Status = PxiIopIoctlv(
			s_hIosSdmc,
			IOCTL_SDIO_SENDCMD,
			2,
			1,
			CmdBuf->Vectors,
			0, 0
		);
		if (Status >= 0) return Status;
		// fallback to ioctl
	}

	return PxiIopIoctl(
		s_hIosSdmc,
		IOCTL_SDIO_SENDCMD,
		&CmdBuf->Command,
		sizeof(IOS_SDMC_COMMAND),
		&CmdBuf->Response,
		sizeof(IOS_SDMC_RESPONSE),
		IOCTL_SWAP_NONE, IOCTL_SWAP_NONE
	);
}

static LONG SdmcpSendCommand(
	SDIO_COMMAND Command,
	SDIO_COMMAND_TYPE Type,
	SDIO_RESPONSE ResponseType,
	ULONG Argument,
	ULONG BlockCount,
	ULONG BlockSize,
	PVOID Buffer,
	PIOS_SDMC_RESPONSE Reply
) {
	// Get the physical address for the DMA.
	ULONG BufferPhys = 0;
	bool IsDma = (Buffer != NULL);
	if (Buffer != NULL) {
		BufferPhys = (ULONG)MEM_VIRTUAL_TO_PHYSICAL(Buffer);
	}

	// Allocate command buffer in IPC RAM.
	PIOS_SDMC_SEND_COMMAND_BUFFER CmdBuf = (PIOS_SDMC_SEND_COMMAND_BUFFER)
		PxiIopAlloc(sizeof(IOS_SDMC_SEND_COMMAND_BUFFER));
	if (CmdBuf == NULL) return -1;

	bool CanTimeoutOnIosSide = true;
	if (
		Command == SDIO_CMD_READBLOCK ||
		Command == SDIO_CMD_READMULTIBLOCK ||
		Command == SDIO_CMD_WRITEBLOCK ||
		Command == SDIO_CMD_WRITEMULTIBLOCK
		) {
		CanTimeoutOnIosSide = false;
	}

	// Toggle the disc LED gpio.
	if (!CanTimeoutOnIosSide)
		SdmcpToggleLed();

	LONG Status = -1;
	do {
		CmdBuf->Command.Command = Command;
		CmdBuf->Command.CommandType = Type;
		CmdBuf->Command.ResponseType = ResponseType;
		CmdBuf->Command.Argument = Argument;
		CmdBuf->Command.BlockCount = BlockCount;
		CmdBuf->Command.BlockSize = BlockSize;
		CmdBuf->Command.Padding0 = CmdBuf->Command.Padding1 = 0;
		CmdBuf->Command.IsDma = IsDma;
		CmdBuf->Command.UserBuffer = BufferPhys;

		if (s_SdmcIsHighCapacity || IsDma) {
			CmdBuf->Vectors[0].Pointer = &CmdBuf->Command;
			CmdBuf->Vectors[0].Length = sizeof(IOS_SDMC_COMMAND);
			CmdBuf->Vectors[1].Pointer = Buffer;
			CmdBuf->Vectors[1].Length = (BlockCount * BlockSize);
			CmdBuf->Vectors[2].Pointer = &CmdBuf->Response;
			CmdBuf->Vectors[2].Length = sizeof(IOS_SDMC_RESPONSE);
		}
		else {
			CmdBuf->Vectors[0].Pointer = NULL;
		}
		Status = SdmcpSendCommandImpl(CmdBuf, CanTimeoutOnIosSide);

		// libogc doesn't check error first...

		if (Reply != NULL)
			memcpy32(Reply, &CmdBuf->Response, sizeof(CmdBuf->Response));
	} while (false);
	PxiIopFree(CmdBuf);
	// Toggle the disc LED gpio.
	if (!CanTimeoutOnIosSide)
		SdmcpToggleLed();
	return Status;
}

static LONG SdmcpSendCommandEx(
	SDIO_COMMAND Command,
	SDIO_COMMAND_TYPE Type,
	SDIO_RESPONSE ResponseType,
	ULONG Argument,
	ULONG BlockCount,
	ULONG BlockSize,
	PVOID Buffer,
	PVOID Reply,
	ULONG ReplyLength
) {
	IOS_SDMC_RESPONSE Response;
	LONG Status = SdmcpSendCommand(Command, Type, ResponseType, Argument, BlockCount, BlockSize, Buffer, &Response);
	if (Status < 0) return Status;
	if (Reply != NULL && ReplyLength <= sizeof(Response)) {
		SdmcpCopyResponse(Reply, &Response, ReplyLength);
		
	}
	return Status;
}

static LONG SdmcpSetClock(ULONG Set) {
	static STACK_ALIGN(ULONG, Clock, 1, 32);
	// we are 32 byte aligned
	// therefore, the correct index is 1
	// MmioWrite32(Clock, Set);
	Clock[1] = Set;
	return PxiIopIoctl(s_hIosSdmc, IOCTL_SDIO_SETCLK, Clock, sizeof(ULONG), NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
}

static LONG SdmcpGetStatus(PULONG Status) {
	static STACK_ALIGN(ULONG, lStatus, 1, 32);
	LONG IopStatus = PxiIopIoctl(s_hIosSdmc, IOCTL_SDIO_GETSTATUS, NULL, 0, lStatus, sizeof(ULONG), IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	if (IopStatus >= 0) *Status = lStatus[1]; // MmioRead32(lStatus)
	return IopStatus;
}

static LONG SdmcpResetCard(void) {
	static STACK_ALIGN(ULONG, lStatus, 1, 32);
	s_SdmcRca = 0;
	LONG Status = PxiIopIoctl(s_hIosSdmc, IOCTL_SDIO_RESETCARD, NULL, 0, lStatus, sizeof(ULONG), IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	if (Status < 0) return Status;
	s_SdmcRca = (USHORT)(LoadToRegister32(lStatus[1]) >> 16);
	return Status;
}

static LONG SdmcpReadRegister(UCHAR Reg, UCHAR Size, PULONG Value) {
	static STACK_ALIGN(IOS_SDMC_MMIO, Mmio, 1, 32);
	static STACK_ALIGN(ULONG, lValue, 1, 32);
	if (Value == NULL) return -1;

	Mmio->Address = Reg;
	Mmio->BlockSize = 0;
	Mmio->BlockCount = 0;
	Mmio->Width = Size;
	Mmio->Value = 0;
	Mmio->IsDma = 0;

	LONG Status = PxiIopIoctl(s_hIosSdmc, IOCTL_SDIO_READHCREG, Mmio, sizeof(*Mmio), lValue, sizeof(*lValue), IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
	if (Status >= 0) *Value = lValue[1];

	return Status;
}

static LONG SdmcpWriteRegister(UCHAR Reg, UCHAR Size, ULONG Value) {
	static STACK_ALIGN(IOS_SDMC_MMIO, Mmio, 1, 32);

	Mmio->Address = Reg;
	Mmio->BlockSize = 0;
	Mmio->BlockCount = 0;
	Mmio->Width = Size;
	Mmio->Value = Value;
	Mmio->IsDma = 0;

	return PxiIopIoctl(s_hIosSdmc, IOCTL_SDIO_WRITEHCREG, Mmio, sizeof(*Mmio), NULL, 0, IOCTL_SWAP_NONE, IOCTL_SWAP_NONE);
}

static bool SdmcpWaitRegister(UCHAR Reg, UCHAR Size, bool Unset, ULONG Mask) {
	ULONG Value;
	LONG Status;

	for (ULONG Try = 0; Try < 10; Try++) {
		if (Try != 0) {
			udelay(10000);
		}
		Status = SdmcpReadRegister(Reg, Size, &Value);
		if (Status < 0) return Status;
		ULONG Masked = Value & Mask;
		if (Unset) {
			if (Masked == 0) return true;
		}
		else {
			if (Masked != 0) return true;
		}
	}
	return false;
}

static LONG SdmcpGetRca(void) {
	static STACK_ALIGN(ULONG, lStatus, 1, 32);
	LONG Status = SdmcpSendCommandEx(SDIO_CMD_SENDRCA, 0, SDIO_RESPONSE_R5, 0, 0, 0, NULL, lStatus, sizeof(*lStatus));
	if (Status < 0) return Status;
	s_SdmcRca = (USHORT)(LoadToRegister32(lStatus[1]) >> 16);
	return Status;
}

static LONG SdmcpSelect(void) {
	return SdmcpSendCommand(SDIO_CMD_SELECT, SDIO_TYPE_CMD, SDIO_RESPONSE_R1B, s_SdmcRca << 16, 0, 0, NULL, NULL);
}

static LONG SdmcpDeSelect(void) {
	return SdmcpSendCommand(SDIO_CMD_DESELECT, SDIO_TYPE_CMD, SDIO_RESPONSE_R1B, 0, 0, 0, NULL, NULL);
}

static LONG SdmcpSetBlockLength(ULONG BlockLen) {
	return SdmcpSendCommand(SDIO_CMD_SETBLOCKLEN, SDIO_TYPE_CMD, SDIO_RESPONSE_R1, BlockLen, 0, 0, NULL, NULL);
}

static LONG SdmcpSetBusWidth(ULONG BusWidth) {
	USHORT Value = 0;
	if (BusWidth == 4) Value = 2;

	LONG Status = SdmcpSendCommand(SDIO_CMD_APPCMD, SDIO_TYPE_CMD, SDIO_RESPONSE_R1, s_SdmcRca << 16, 0, 0, NULL, NULL);
	if (Status < 0) return Status;

	return SdmcpSendCommand(SDIO_ACMD_SETBUSWIDTH, SDIO_TYPE_CMD, SDIO_RESPONSE_R1, Value, 0, 0, NULL, NULL);
}

static LONG SdmcpIoSetBusWidth(ULONG BusWidth) {
	ULONG Reg;
	LONG Status = SdmcpReadRegister(SDIOHCR_HOSTCONTROL, 1, &Reg);
	if (Status < 0) return Status;

	Reg &= 0xff;
	Reg &= ~SDIOHCR_HOSTCONTROL_4BIT;
	if (BusWidth == 4) Reg |= SDIOHCR_HOSTCONTROL_4BIT;

	return SdmcpWriteRegister(SDIOHCR_HOSTCONTROL, 1, Reg);
}

static LONG SdmcpGetCsd(void) {
	LONG Status = SdmcpSendCommandEx(SDIO_CMD_SENDCSD, SDIO_TYPE_CMD, SDIO_RESPONSE_R2, s_SdmcRca << 16, 0, 0, NULL, s_SdmcCsd, sizeof(s_SdmcCsd));
	if (Status >= 0) endian_swap64(s_SdmcCsd, s_SdmcCsd, sizeof(s_SdmcCsd));
	return Status;
}

static LONG SdmcpGetCid(void) {
	LONG Status = SdmcpSendCommandEx(SDIO_CMD_ALL_SENDCID, 0, SDIO_RESPONSE_R2, s_SdmcRca << 16, 0, 0, NULL, s_SdmcCid, sizeof(s_SdmcCid));
	if (Status >= 0) endian_swap64(s_SdmcCid, s_SdmcCid, sizeof(s_SdmcCid));
	return Status;
}

static LONG SdmcpIopReOpen(void) {
	LONG Status = -1;
	if (s_hIosSdmc != IOS_HANDLE_INVALID) {
		Status = PxiIopClose(s_hIosSdmc);
		if (Status < 0) return Status;
		s_hIosSdmc = IOS_HANDLE_INVALID;
	}
	return PxiIopOpen("/dev/sdio/slot0", IOSOPEN_READ, &s_hIosSdmc);
}

static LONG SdmcpInitIoEx(void) {
	LONG Status = -1;
	IOS_SDMC_RESPONSE Response;
	do {
		// Reset the sdmmc block.
		Status = SdmcpWriteRegister(SDIOHCR_SOFTWARERESET, 1, 7);
		if (Status < 0) break;
		Status = SdmcpWaitRegister(SDIOHCR_SOFTWARERESET, 1, true, 7);
		if (Status < 0) break;

		// Initialise interrupts.
		Status = SdmcpWriteRegister(0x34, 4, 0x13f00c3);
		if (Status < 0) break;
		Status = SdmcpWriteRegister(0x38, 4, 0x13f00c3);
		if (Status < 0) break;

		// Enable power.
		s_SdmcIsHighCapacity = true;
		Status = SdmcpWriteRegister(SDIOHCR_POWERCONTROL, 1, 0xe);
		if (Status < 0) break;
		Status = SdmcpWriteRegister(SDIOHCR_POWERCONTROL, 1, 0xf);
		if (Status < 0) break;

		// Enable internal clock.
		Status = SdmcpWriteRegister(SDIOHCR_CLOCKCONTROL, 2, 0);
		if (Status < 0) break;
		Status = SdmcpWriteRegister(SDIOHCR_CLOCKCONTROL, 2, 0x101);
		if (Status < 0) break;
		// Wait until it gets stable.
		Status = SdmcpWaitRegister(SDIOHCR_CLOCKCONTROL, 2, false, 2);
		if (Status < 0) break;
		// Enable SD clock.
		Status = SdmcpWriteRegister(SDIOHCR_CLOCKCONTROL, 2, 0x107);
		if (Status < 0) break;

		// Setup timeout.
		Status = SdmcpWriteRegister(SDIOHCR_TIMEOUTCONTROL, 1, SDIO_DEFAULT_TIMEOUT);
		if (Status < 0) break;

		// SDHC init
		Status = SdmcpSendCommand(
			SDIO_CMD_GOIDLE,
			SDIO_TYPE_NONE,
			SDIO_RESPONSE_NONE,
			0, 0, 0, NULL, NULL
		);
		if (Status < 0) break;
		Status = SdmcpSendCommand(
			SDIO_CMD_SENDIFCOND,
			SDIO_TYPE_NONE,
			SDIO_RESPONSE_R6,
			0x1aa,
			0, 0, NULL, &Response
		);
		if (Status < 0) break;
		if ((Response.Field0 & 0xFF) != 0xAA) break;

		bool Success = false;
		for (ULONG Try = 0; Try < 10; Try++) {
			if (Try != 0) {
				udelay(10000);
			}
			Status = SdmcpSendCommand(
				SDIO_CMD_APPCMD,
				SDIO_TYPE_CMD,
				SDIO_RESPONSE_R1,
				0, 0, 0, NULL, NULL
			);
			if (Status < 0) break;
			Status = SdmcpSendCommand(
				SDIO_ACMD_SENDOPCOND,
				SDIO_TYPE_NONE,
				SDIO_RESPONSE_R3,
				0x40300000,
				0, 0, NULL, &Response
			);
			if (Status < 0) break;
			if ((Response.Field0 & BIT(31)) != 0) {
				Success = true;
				break;
			}
		}

		if (Success == false) break;

		// BUGBUG: SDv2 cards which are not high capacity won't work
		// ...but how many of those actually exist?
		s_SdmcIsHighCapacity = (Response.Field0 & BIT(30)) != 0;

		Status = SdmcpGetCid();
		if (Status < 0) break;
		Status = SdmcpGetRca();
		if (Status < 0) break;
		return 0;
	} while (false);

	SdmcpWriteRegister(SDIOHCR_SOFTWARERESET, 1, 7);
	SdmcpWaitRegister(SDIOHCR_SOFTWARERESET, 1, true, 7);

	SdmcpIopReOpen();
	return Status;
}

static LONG SdmcpInitIo(void) {
	LONG Status = SdmcpResetCard();
	if (Status < 0) return Status;
	ULONG SdmcStatus;
	Status = SdmcpGetStatus(&SdmcStatus);
	if (Status < 0) return Status;

	if ((SdmcStatus & SDIO_STATUS_CARD_INSERTED) == 0)
		return -1;

	if ((SdmcStatus & SDIO_TYPE_MEMORY) == 0) {
		// IOS doesn't know what this sdmmc device is.
		// Clean up by closing and reopening the handle.
		Status = SdmcpIopReOpen();
		if (Status < 0) return Status;
		Status = SdmcpInitIoEx();
		if (Status < 0) return Status;
	}
	else {
		s_SdmcIsHighCapacity = (SdmcStatus & SDIO_TYPE_SDHC) != 0;
	}

	Status = SdmcpIoSetBusWidth(4);
	if (Status < 0) return Status;
	Status = SdmcpSetClock(1);
	if (Status < 0) return Status;
	Status = SdmcpSelect();
	if (Status < 0) return Status;
	Status = SdmcpSetBlockLength(SDMC_SECTOR_SIZE);
	if (Status >= 0) {
		Status = SdmcpSetBusWidth(4);
	}
	SdmcpDeSelect();
	if (Status < 0) return Status;

	s_SdmcIsInitialised = true;
	return 0;
}

bool SdmcFinalise(void) {
	if (s_hIosSdmc != IOS_HANDLE_INVALID) {
		LONG Status = PxiIopClose(s_hIosSdmc);
		if (Status < 0) return false;
		s_hIosSdmc = IOS_HANDLE_INVALID;
	}
	s_SdmcIsInitialised = false;
	return true;
}

bool SdmcStartup(void) {
	if (s_SdmcIsInitialised != false) return true;

	LONG Status = SdmcpIopReOpen();
	if (Status < 0) return false;

	Status = SdmcpInitIo();
	if (Status < 0) return false;

	return true;
}

bool SdmcIsMounted(void) { return s_SdmcIsInitialised;  }

bool SdmcIsWriteProtected(void) {
	if (!s_SdmcIsInitialised) return false;

	ULONG CardStatus;
	LONG Status = SdmcpGetStatus(&CardStatus);
	if (Status < 0) {
		return false;
	}

	if ((CardStatus & SDIO_STATUS_CARD_INSERTED) == 0) return false;
	return ((CardStatus & SDIO_STATUS_CARD_WRITEPROT) != 0);
}

ULONG SdmcSectorCount(void) {
	if (!s_SdmcIsInitialised) return 0;

	if (SdmcpGetCsd() < 0) return 0;

	if ((s_SdmcCsd[0] >> 6) == 1) {	/* SDC ver 2.00 */
		ULONG cs = s_SdmcCsd[9] + ((USHORT)s_SdmcCsd[8] << 8) + ((ULONG)(s_SdmcCsd[7] & 63) << 16) + 1;
		return cs << 10;
	}
	else {					/* SDC ver 1.XX or MMC */
		UCHAR n = (s_SdmcCsd[5] & 15) + ((s_SdmcCsd[10] & 128) >> 7) + ((s_SdmcCsd[9] & 3) << 1) + 2;
		ULONG cs = (s_SdmcCsd[8] >> 6) + ((USHORT)s_SdmcCsd[7] << 2) + ((USHORT)(s_SdmcCsd[6] & 3) << 10) + 1;
		return cs << (n - 9);
	}
}

ULONG SdmcReadSectors(ULONG Sector, ULONG NumSector, PVOID Buffer) {
	if (!s_SdmcIsInitialised) {
		return 0;
	}
	if (Buffer == NULL) {
		return 0;
	}

	LONG Status = SdmcpSelect();
	if (Status < 0) return 0;

	// Read up to 16 sectors at a time, into the DMA buffer.
	PUCHAR Pointer = (PUCHAR)Buffer;
	ULONG SectorsTransferred = 0;
	while (NumSector != 0) {
		ULONG Offset = Sector;
		if (s_SdmcIsHighCapacity == false) Offset *= SDMC_SECTOR_SIZE;
		ULONG IoCount = SDMC_SECTORS_IN_PAGE;
		if (NumSector < SDMC_SECTORS_IN_PAGE) IoCount = NumSector;

		Status = SdmcpSendCommand(
			SDIO_CMD_READMULTIBLOCK,
			SDIO_TYPE_CMD,
			SDIO_RESPONSE_R1,
			Offset,
			IoCount,
			SDMC_SECTOR_SIZE,
			s_SdmcDmaBuffer,
			NULL
		);
		if (Status < 0) {
			break;
		}

		// Swap64 from the DMA buffer into the DMA buffer.
		if (((ULONG)Pointer & 7) == 0) {
			// Pointer is 64-bit aligned, swap from the DMA buffer copying into the buffer.
			endian_swap64(Pointer, s_SdmcDmaBuffer, SDMC_SECTOR_SIZE * IoCount);
		}
		else {
			endian_swap64(s_SdmcDmaBuffer, s_SdmcDmaBuffer, SDMC_SECTOR_SIZE * IoCount);
			memcpy(Pointer, s_SdmcDmaBuffer, SDMC_SECTOR_SIZE * IoCount);
		}
		Pointer += SDMC_SECTOR_SIZE * IoCount;
		Sector += IoCount;
		NumSector -= IoCount;
		SectorsTransferred += IoCount;
	}

	SdmcpDeSelect();
	return SectorsTransferred;
}

ULONG SdmcWriteSectors(ULONG Sector, ULONG NumSector, const void* Buffer) {
	if (!s_SdmcIsInitialised) return 0;
	if (Buffer == NULL) return 0;

	LONG Status = SdmcpSelect();
	if (Status < 0) return 0;

	PUCHAR Pointer = (PUCHAR)Buffer;
	ULONG SectorsTransferred = 0;
	while (NumSector != 0) {
		ULONG Offset = Sector;
		if (s_SdmcIsHighCapacity == false) Offset *= SDMC_SECTOR_SIZE;
		ULONG IoCount = SDMC_SECTORS_IN_PAGE;
		if (NumSector < SDMC_SECTORS_IN_PAGE) IoCount = NumSector;

		if (((ULONG)Pointer & 7) == 0) {
			// Pointer is 64-bit aligned, swap from the pointer directly into the DMA buffer
			endian_swap64(s_SdmcDmaBuffer, Pointer, SDMC_SECTOR_SIZE * IoCount);
		}
		else {
			// Not 64-bit aligned, copy then swap.
			memcpy(s_SdmcDmaBuffer, Pointer, SDMC_SECTOR_SIZE * IoCount);
			endian_swap64(s_SdmcDmaBuffer, s_SdmcDmaBuffer, SDMC_SECTOR_SIZE * IoCount);
		}

		Status = SdmcpSendCommand(
			SDIO_CMD_WRITEMULTIBLOCK,
			SDIO_TYPE_CMD,
			SDIO_RESPONSE_R1,
			Offset,
			IoCount,
			SDMC_SECTOR_SIZE,
			s_SdmcDmaBuffer,
			NULL
		);
		if (Status < 0) break;
		Pointer += SDMC_SECTOR_SIZE * IoCount;
		Sector += IoCount;
		NumSector -= IoCount;
		SectorsTransferred += IoCount;
	}

	SdmcpDeSelect();
	return SectorsTransferred;
}
