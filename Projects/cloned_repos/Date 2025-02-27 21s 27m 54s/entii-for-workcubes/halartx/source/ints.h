#pragma once

typedef enum {
#define INTERRUPT_DEFINE(key,val) VECTOR_##key = val,
#include "ints.inc"
#undef INTERRUPT_DEFINE
} INTERRUPT_VECTOR;

typedef enum {
#define INTERRUPT_DEFINE(key,val) INTERRUPT_##key = BIT(val),
#include "ints.inc"
#undef INTERRUPT_DEFINE
} INTERRUPT_SOURCE;

typedef struct {
	volatile ULONG Cause;
	volatile ULONG Mask;
	volatile ULONG Unused[4];
	volatile ULONG CpAbort;
} PI_INTERRUPT_REGS, *PPI_INTERRUPT_REGS;

typedef struct {
	volatile ULONG Cause;
	volatile ULONG Mask;
} PI_INTERRUPT_CPU_LATTE, *PPI_INTERRUPT_CPU_LATTE;

typedef struct {
	PI_INTERRUPT_CPU_LATTE Global; // maybe?
	volatile ULONG Unused[(0x40 - 0x8) / 4];
	volatile ULONG WriteGather[12];
	volatile ULONG Unused2[2];
	PI_INTERRUPT_CPU_LATTE Cpu[3];
} PI_INTERRUPT_REGS_LATTE, *PPI_INTERRUPT_REGS_LATTE;

enum {
	PI_INTERRUPT_REGS_BASE = 0x0c003000,
	PI_INTERRUPT_REGS_LATTE_BASE = 0x0c000000,
};

typedef BOOLEAN (*PSECONDARY_DISPATCH)(
	PVOID InterruptRoutine,
	PVOID ServiceContext,
	PVOID TrapFrame
);

BOOLEAN HalpEnableDeviceInterruptHandler(
    IN PKINTERRUPT Interrupt,
    IN PKSERVICE_ROUTINE ServiceRoutine,
    IN PVOID ServiceContext,
    IN PKSPIN_LOCK SpinLock OPTIONAL,
    IN INTERRUPT_VECTOR Vector,
    IN KINTERRUPT_MODE InterruptMode,
    IN BOOLEAN ShareVector,
    IN CCHAR ProcessorNumber,
    IN BOOLEAN FloatingSave,
    IN UCHAR    ReportFlags
);

void HalpDisableDeviceInterruptHandler(INTERRUPT_VECTOR Vector);

KIRQL HalpRaiseDeviceIrql(INTERRUPT_VECTOR Vector);
