// Implements RTC related functions
#include "halp.h"
#include "exiapi.h"
#include "exi.h"

extern PEXI_REGISTERS HalpExiRegs; // implemented by kd.c

// exi channel 0 device 1 is the SPI RTC (ROM+RTC on systems prior to Cafe)
#define EXI_CHAN0 Channel[0]

enum {
	UNIX_TIME_2000 = 946684800, // seconds from 1 jan 1970 to 1 jan 2000
	UNIX_TIME_1980 = 315532800, // seconds from 1 jan 1970 to 1 jan 1980
};

#if 0
static ULONG RtcSendAndReceive(ULONG Data) {
	// Send the command and wait for response.
	MmioWriteBase32(MMIO_OFFSET(HalpExiRegs, EXI_CHAN0.Parameter), 0x130);
	MmioWriteBase32(MMIO_OFFSET(HalpExiRegs, EXI_CHAN0.Data), Data);
	MmioWriteBase32(MMIO_OFFSET(HalpExiRegs, EXI_CHAN0.Transfer), 0x39);
	while ((MmioReadBase32(MMIO_OFFSET(HalpExiRegs, EXI_CHAN0.Transfer)) & 1) != 0) { }
	ULONG Recv = MmioReadBase32(MMIO_OFFSET(HalpExiRegs, EXI_CHAN0.Data));
	MmioWriteBase32(MMIO_OFFSET(HalpExiRegs, EXI_CHAN0.Parameter), 0);
	return Recv;
}
#endif

static ULONG RtcGetTime(void) {
	//return RtcSendAndReceive(0x20000000);
	
	NTSTATUS Status = HalExiSelectDevice(0, 1, EXI_CLOCK_6_7, FALSE);
	if (!NT_SUCCESS(Status)) return 0;

	ULONG Value = 0;
	do {
		Status = HalExiTransferImmediate(0, 0x20000000, 4, EXI_TRANSFER_WRITE, NULL);
		if (!NT_SUCCESS(Status)) break;
		Status = HalExiTransferImmediate(0, 0, 4, EXI_TRANSFER_READ, &Value);
		if (!NT_SUCCESS(Status)) break;
	} while (0);

	HalExiUnselectDevice(0);
	return Value;
}

static ULONG HalpRtcGetTimeImpl(void) {
	while (TRUE) {
		ULONG Time1 = RtcGetTime();
		ULONG Time2 = RtcGetTime();
		if (Time1 == 0 || Time2 == 0) return 0;
		if (Time1 != Time2) continue;
		return Time1 + (ULONG)RUNTIME_BLOCK[RUNTIME_RTC_BIAS];
	}
}

typedef struct _RTC_LOCK_CONTEXT {
	ULONG Time;
	KEVENT Event;
} RTC_LOCK_CONTEXT, *PRTC_LOCK_CONTEXT;

static EXI_LOCK_ACTION HalpRtcGetTimeLocked(ULONG channel, PVOID context) {
	PRTC_LOCK_CONTEXT RtcCtx = (PRTC_LOCK_CONTEXT)context;
	RtcCtx->Time = HalpRtcGetTimeImpl();
	KeSetEvent(&RtcCtx->Event, (KPRIORITY)0, FALSE);
	return ExiUnlock;
}

ULONG HalpRtcGetTime(void) {
	RTC_LOCK_CONTEXT Context = {0};
	KeInitializeEvent(&Context.Event, NotificationEvent, FALSE);
	NTSTATUS Status = HalExiLockNonpaged(0, HalpRtcGetTimeLocked, &Context);
	if (!NT_SUCCESS(Status)) return 0;
	KeWaitForSingleObject(&Context.Event, Executive, KernelMode, FALSE, NULL);
	return Context.Time;
}

BOOLEAN HalQueryRealTimeClock (OUT PTIME_FIELDS TimeFields) {
	ULONG Time = HalpRtcGetTime();
	if (Time == 0) return FALSE;
	// convert seconds from 2000 to seconds from 1980
	// yes, there's RtlSecondsSince1970ToTime
	// use the later epoch for a later time problem lol
	ULONG SecondsSince1980 = Time + (UNIX_TIME_2000 - UNIX_TIME_1980);
	LARGE_INTEGER NtTime;
	RtlSecondsSince1980ToTime(SecondsSince1980, &NtTime);
	RtlTimeToTimeFields(&NtTime, TimeFields);
	return TRUE;
}

BOOLEAN HalSetRealTimeClock (IN PTIME_FIELDS TimeFields) {
	// don't touch the RTC.
	return FALSE;
}