// SI driver
#define DEVL 1
#include <ntddk.h>
#include "runtime.h"
#include "si.h"
#include "sireg.h"

typedef struct _SI_CHANNEL_DATA {
	UCHAR Value[SI_POLL_LENGTH];
} SI_CHANNEL_DATA, *PSI_CHANNEL_DATA;

typedef struct _SI_TRANSFER_DESCRIPTOR {
	LIST_ENTRY ListEntry;
	ULONG Channel;
	PVOID Output;
	ULONG OutputLength;
	PVOID Input;
	ULONG InputLength;
	SI_TRANSFER_CALLBACK Callback;
	ULONG Success;
	PRKEVENT Event;
} SI_TRANSFER_DESCRIPTOR, *PSI_TRANSFER_DESCRIPTOR;

typedef struct _SI_TRANSFER_DESCRIPTOR_SYNC {
	SI_TRANSFER_DESCRIPTOR Base;
	KEVENT Event;
} SI_TRANSFER_DESCRIPTOR_SYNC, *PSI_TRANSFER_DESCRIPTOR_SYNC;

enum {
	VECTOR_SI = 3
};

static PSI_REGISTERS SiRegisters;
static PKINTERRUPT s_Interrupt;
static KIRQL s_DeviceIrql;

static LIST_ENTRY s_DescriptorList;
static KSPIN_LOCK s_DescriptorListLock;

static KDPC s_DpcTransferComplete[SI_CHANNEL_COUNT],
	s_DpcReadStatus[SI_CHANNEL_COUNT];

static SI_POLL_CALLBACK s_PollCallback[SI_CHANNEL_COUNT];

static SI_CHANNEL_DATA s_ChannelData[SI_CHANNEL_COUNT];

static void SipAddTransfer(PSI_TRANSFER_DESCRIPTOR Entry) {
	ExInterlockedInsertHeadList(&s_DescriptorList, &Entry->ListEntry, &s_DescriptorListLock);
}

static void SipRemoveTransfer(void) {
	ExInterlockedRemoveHeadList(&s_DescriptorList, &s_DescriptorListLock);
}

static void SipClearTransferCompleteInterrupt(ULONG Value) {
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER Comcs;
	Comcs.Value = Value;
	Comcs.TransferStart = 0;
	Comcs.TransferCompleteInterruptStatus = 1;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus), Comcs.Value);
}

static BOOLEAN SipSiInterrupt(PKINTERRUPT InterruptRoutine, PVOID ServiceContext) {
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER Comcs;
	Comcs.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus));
	if (Comcs.TransferCompleteInterruptStatus == 1 && Comcs.TransferCompleteInterruptMask == 1) {
		// Clear the interrupt.
		SipClearTransferCompleteInterrupt(Comcs.Value);
		// Fire the DPC.
		BOOLEAN AlreadyQueued = FALSE;
		for (ULONG i = 0; i < SI_CHANNEL_COUNT; i++) {
			AlreadyQueued = KeInsertQueueDpc(&s_DpcTransferComplete[i], (PVOID)(ULONG)Comcs.Channel, (PVOID)(ULONG)Comcs.CommunicationError);
			if (AlreadyQueued) break;
		}
	}
	if (Comcs.ReadStatusInterruptStatus == 1 && Comcs.ReadStatusInterruptMask == 1) {
		BOOLEAN ChannelChanged[SI_CHANNEL_COUNT];
		
		// If a channel got updated, update its data and fire its DPC.
		SI_STATUS_REGISTER Status;
		Status.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, Status));
		for (ULONG channel = 0; channel < SI_CHANNEL_COUNT; channel++) {
			ULONG chanInvert = (SI_CHANNEL_COUNT - 1) - channel;
			if (!Status.Channels[chanInvert].ReadStatus) continue;
			// Read the input registers.
			// Reading the high register clears the interrupt, but reading the high register also locks the values until the low register is read.
			SI_CHANNEL_RESPONSE_HIGH_REGISTER High;
			SI_CHANNEL_RESPONSE_LOW_REGISTER Low;
			High.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, Channel[channel].ResponseHigh));
			Low.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, Channel[channel].ResponseLow));
			s_ChannelData[channel].Value[0] = High.Data0;
			s_ChannelData[channel].Value[1] = High.Data1;
			s_ChannelData[channel].Value[2] = High.Data2;
			s_ChannelData[channel].Value[3] = High.Data3;
			s_ChannelData[channel].Value[4] = Low.Data0;
			s_ChannelData[channel].Value[5] = Low.Data1;
			s_ChannelData[channel].Value[6] = Low.Data2;
			s_ChannelData[channel].Value[7] = Low.Data3;
			// Fire the DPC.
			KeInsertQueueDpc(&s_DpcReadStatus[channel], (PVOID)channel, NULL);
		}
	}
}

static void SipPollDpc(PKDPC Dpc, PVOID Unused, PVOID ChannelArg, PVOID Unused2) {
	ULONG channel = (ULONG)ChannelArg;
	if (s_PollCallback[channel] != NULL)
		s_PollCallback[channel](channel, s_ChannelData[channel].Value);
}

static __attribute__((const)) inline ULONG SipGetPhysBase(ULONG SystemType)  {
	ULONG base = SI_REGISTER_BASE_FLIPPER;
	if (SystemType != ARTX_SYSTEM_FLIPPER) base += SI_REGISTER_OFFSET_VEGAS;
	return base;
}

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

static void SipBufferCopy(PVOID buffer, ULONG length, BOOLEAN writeIo) {
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

static void SiNextTransferAsync(PSI_TRANSFER_DESCRIPTOR Descriptor) {
	ULONG channel = Descriptor->Channel;
	PVOID bufRead = Descriptor->Input;
	PVOID bufWrite = Descriptor->Output;
	ULONG lenWrite = Descriptor->OutputLength;
	ULONG lenRead = Descriptor->InputLength;
	
	// Invert the channel number to get the correct index into the array.
	// Could use MmioWrite8/etc, but this is one instruction.
	ULONG chanInvert = (SI_CHANNEL_COUNT - 1) - channel;

	KIRQL OldIrql;
	KeRaiseIrql(s_DeviceIrql, &OldIrql);
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
	SipBufferCopy(bufWrite, lenWrite, TRUE);

	if (lenWrite == SI_MAX_TRANSFER_LENGTH) lenWrite = 0;
	if (lenRead == SI_MAX_TRANSFER_LENGTH) lenRead = 0;

	// Configure and start the transfer
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER Comcs;
	Comcs.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus));
	Comcs.TransferCompleteInterruptStatus = 1; // Clear interrupt
	Comcs.TransferCompleteInterruptMask = 1; // Enable interrupt (for asynchronous transfer)
	Comcs.OutputLength = lenWrite;
	Comcs.InputLength = lenRead;
	Comcs.Channel = channel;
	Comcs.TransferStart = 1;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus), Comcs.Value);
	KeLowerIrql(OldIrql);
}

static void SipTransferCompleteDpc(PKDPC Dpc, PVOID Unused, PVOID ChannelArg, PVOID ErrorArg) {
	ULONG Channel = (ULONG)ChannelArg;
	
	// Pull the descriptor off the top of the list.
	if (IsListEmpty(&s_DescriptorList)) return;
	PSI_TRANSFER_DESCRIPTOR Descriptor = (PSI_TRANSFER_DESCRIPTOR) s_DescriptorList.Flink;
	SipRemoveTransfer();
	
	if (Channel != Descriptor->Channel) {
		// What happened here?
		return;
	}
	
	// Copy the data from the SI I/O buffer to the output buffer
	SipBufferCopy(Descriptor->Input, Descriptor->InputLength, FALSE);
	
	BOOLEAN Success = ((ULONG)ErrorArg) == 0;
	
	// Call the callback, or set the event
	if (Descriptor->Event != NULL) {
		Descriptor->Success = Success;
		KeSetEvent(Descriptor->Event, (KPRIORITY)0, FALSE);
	} else {
		if (Descriptor->Callback != NULL)
			Descriptor->Callback(Channel, Success, Descriptor->Input, Descriptor->InputLength);
		// Free the descriptor.
		ExFreePool(Descriptor);
	}
	
	if (IsListEmpty(&s_DescriptorList)) return;
	
	Descriptor = (PSI_TRANSFER_DESCRIPTOR) s_DescriptorList.Flink;
	SiNextTransferAsync(Descriptor);
}

static void SipTransferInitDescriptor(PSI_TRANSFER_DESCRIPTOR Descriptor, ULONG channel, PVOID bufWrite, ULONG lenWrite, PVOID bufRead, ULONG lenRead, SI_TRANSFER_CALLBACK callback) {
	RtlZeroMemory(Descriptor, sizeof(*Descriptor));
	Descriptor->Channel = channel;
	Descriptor->Output = bufWrite;
	Descriptor->OutputLength = lenWrite;
	Descriptor->Input = bufRead;
	Descriptor->InputLength = lenRead;
	Descriptor->Callback = callback;
}

NTSTATUS SiTransferAsync(ULONG channel, PVOID bufWrite, ULONG lenWrite, PVOID bufRead, ULONG lenRead, SI_TRANSFER_CALLBACK callback) {
	if (channel >= SI_CHANNEL_COUNT) return STATUS_INVALID_PARAMETER;
	if (lenWrite > SI_MAX_TRANSFER_LENGTH || lenRead > SI_MAX_TRANSFER_LENGTH) return STATUS_INVALID_PARAMETER;
	if (lenWrite == 0 || lenRead == 0) return STATUS_INVALID_PARAMETER;
	if (callback == NULL) return STATUS_INVALID_PARAMETER;
	
	PSI_TRANSFER_DESCRIPTOR Descriptor = ExAllocatePool(NonPagedPool, sizeof(SI_TRANSFER_DESCRIPTOR));
	if (Descriptor == NULL) return STATUS_NO_MEMORY;
	
	SipTransferInitDescriptor(Descriptor, channel, bufWrite, lenWrite, bufRead, lenRead, callback); 
	
	BOOLEAN Empty = IsListEmpty(&s_DescriptorList);
	SipAddTransfer(Descriptor);
	
	if (Empty) {
		SiNextTransferAsync(Descriptor);
		return STATUS_SUCCESS;
	}
	
	return STATUS_PENDING;
}


NTSTATUS SiTransferSync(ULONG channel, PVOID bufWrite, ULONG lenWrite, PVOID bufRead, ULONG lenRead) {
	if (channel >= SI_CHANNEL_COUNT) return STATUS_INVALID_PARAMETER;
	if (lenWrite > SI_MAX_TRANSFER_LENGTH || lenRead > SI_MAX_TRANSFER_LENGTH) return STATUS_INVALID_PARAMETER;
	if (lenWrite == 0 || lenRead == 0) return STATUS_INVALID_PARAMETER;
	
	// Allocate a sync descriptor, with kevent, add it to the end of the list, and wait on the event.
	PSI_TRANSFER_DESCRIPTOR_SYNC DescriptorSync = ExAllocatePool(NonPagedPool, sizeof(SI_TRANSFER_DESCRIPTOR_SYNC));
	if (DescriptorSync == NULL) return STATUS_NO_MEMORY;
	
	// Initailise the descriptor and kevent
	SipTransferInitDescriptor(&DescriptorSync->Base, channel, bufWrite, lenWrite, bufRead, lenRead, NULL);
	KeInitializeEvent(&DescriptorSync->Event, SynchronizationEvent, FALSE);
	DescriptorSync->Base.Event = &DescriptorSync->Event;
	
	// Add to end of list.
	BOOLEAN Empty = IsListEmpty(&s_DescriptorList);
	SipAddTransfer(&DescriptorSync->Base);
	if (Empty) {
		SiNextTransferAsync(&DescriptorSync->Base);
	}
	
	// Wait for the transfer.
	KeWaitForSingleObject( &DescriptorSync->Event, Executive, KernelMode, FALSE, NULL );
	
	// Free the descriptor and return success.
	BOOLEAN Success = (DescriptorSync->Base.Success != 0);
	ExFreePool(DescriptorSync);
	return Success ? STATUS_SUCCESS : STATUS_ADAPTER_HARDWARE_ERROR;
}

NTSTATUS SiPollSetCallback(ULONG channel, SI_POLL_CALLBACK callback) {
	if (channel >= SI_CHANNEL_COUNT) return STATUS_INVALID_PARAMETER;
	s_PollCallback[channel] = callback;
	return STATUS_SUCCESS;
}

NTSTATUS SiTransferPoll(ULONG channels, ULONG data, ULONG length) {
	if (channels == 0 || channels >= BIT(SI_CHANNEL_COUNT)) return STATUS_INVALID_PARAMETER;
	if (length > 3) return STATUS_INVALID_PARAMETER;
	if (length != 0) data <<= (3 - length) * 8;
	
	KIRQL OldIrql;
	KeRaiseIrql(s_DeviceIrql, &OldIrql);
	SI_POLL_REGISTER Poll;
	Poll.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, Poll));
	for (ULONG channel = 0; channel < SI_CHANNEL_COUNT; channel++) {
		if ((channels & BIT(channel)) == 0) continue;
		ULONG chanInvert = (SI_CHANNEL_COUNT - 1) - channel;
		MmioWriteBase32(MMIO_OFFSET(SiRegisters, Channel[channel].Command), data);
		if (channel == 0) Poll.Enable0 = Poll.VblankCopy0 = (length != 0);
		else if (channel == 1) Poll.Enable1 = Poll.VblankCopy1 = (length != 0);
		else if (channel == 2) Poll.Enable2 = Poll.VblankCopy2 = (length != 0);
		else if (channel == 3) Poll.Enable3 = Poll.VblankCopy3 = (length != 0);
	}
	SI_STATUS_REGISTER Status;
	Status.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, Status));
	Status.WriteAll = 1;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, Status), Status.Value);
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, Poll), Poll.Value);
	KeLowerIrql(OldIrql);
	return STATUS_SUCCESS;
}

void SiTogglePoll(BOOLEAN value) {
	KIRQL OldIrql;
	KeRaiseIrql(s_DeviceIrql, &OldIrql);
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER Comcs;
	Comcs.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus));
	Comcs.ReadStatusInterruptMask = value;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus), Comcs.Value);
	KeLowerIrql(OldIrql);
}

static JOYBUS_DEVICE_TYPE SipGetDeviceType(ULONG channel, UCHAR command) {
	JOYBUS_DEVICE_TYPE Type;
	Type.Value = 0;
	if (!NT_SUCCESS(SiTransferByteSync(channel, command, &Type.Value, sizeof(Type) - 1))) {
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

NTSTATUS SiInit(void) {
	
	InitializeListHead(&s_DescriptorList);
	KeInitializeSpinLock(&s_DescriptorListLock);
	
	// Initialise the DPCs.
	for (ULONG i = 0; i < SI_CHANNEL_COUNT; i++) {
		KeInitializeDpc(&s_DpcTransferComplete[i], SipTransferCompleteDpc, NULL);
		KeInitializeDpc(&s_DpcReadStatus[i], SipPollDpc, NULL);
	}
	
	
	// Map SI registers.
	ULONG SystemType = (ULONG)RUNTIME_BLOCK[RUNTIME_SYSTEM_TYPE];
	PHYSICAL_ADDRESS SiPhys;
	SiPhys.HighPart = 0;
	SiPhys.LowPart = SipGetPhysBase(SystemType);
	SiRegisters = MmMapIoSpace(SiPhys, sizeof(SI_REGISTERS), MmNonCached);
	if (SiRegisters == NULL) return STATUS_NO_MEMORY;
	KAFFINITY Affinity;
	ULONG InterruptVector = HalGetInterruptVector(Internal, 0, 0, VECTOR_SI, &s_DeviceIrql, &Affinity);
	
	
	// BUGBUG: Assume that poll register contains the value that any loader wrote to it :3
	
	// Wait for any transfer to complete
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER Comcs;
	do {
		Comcs.Value = MmioReadBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus));
	} while (Comcs.TransferStart);

	// Disable interrupts and acknowledge transfer complete interrupt
	Comcs.Value = 0;
	Comcs.TransferCompleteInterruptStatus = 1;
	MmioWriteBase32(MMIO_OFFSET(SiRegisters, CommunicationControlStatus), Comcs.Value);
	

	NTSTATUS Status = IoConnectInterrupt(&s_Interrupt, SipSiInterrupt, NULL, NULL, InterruptVector, s_DeviceIrql, s_DeviceIrql, LevelSensitive, FALSE, Affinity, FALSE);
	if (!NT_SUCCESS(Status)) {
		MmUnmapIoSpace(SiRegisters, sizeof(SI_REGISTERS));
		return Status;
	}
	
}