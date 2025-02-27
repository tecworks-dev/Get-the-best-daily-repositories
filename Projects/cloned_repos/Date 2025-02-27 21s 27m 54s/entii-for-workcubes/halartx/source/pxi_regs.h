#pragma once

typedef enum _PXI_CONTROL {
	PXI_REQ_SEND = BIT(0), /// < Send request to IOP.
	PXI_REQ_ACK = BIT(1), /// < IOP acknowledged request. Write to clear.
	PXI_RES_SENT = BIT(2), /// < IOP responded. Write to clear.
	PXI_RES_ACK = BIT(3), /// < Acknowledge response to IOP.
	PXI_RES_SENT_INT = BIT(4), /// < Raise interrupt when IOP responds.
	PXI_REQ_ACK_INT = BIT(5), /// < Raise interrupt when IOP acks.
	
	PXI_BITS_PRESERVE = PXI_RES_SENT_INT | PXI_REQ_ACK_INT /// < Bits to preserve when clearing interrupt statuses.
} PXI_CONTROL;

typedef struct _PXI_CORE_REGISTERS {
	ULONG Message;
	ULONG Control; // PXI_CONTROL
} PXI_CORE_REGISTERS, *PPXI_CORE_REGISTERS;

typedef struct _PXI_REGISTERS {
	PXI_CORE_REGISTERS Request, Response;
	ULONG Reserved[(0x30 - 0x10) / sizeof(ULONG)];
	ULONG InterruptCause;
	ULONG InterruptMask;
} PXI_REGISTERS, *PPXI_REGISTERS;

_Static_assert(sizeof(PXI_REGISTERS) == 0x38);

enum {
	VEGAS_INTERRUPT_GPIO = BIT(10),
	VEGAS_INTERRUPT_PXI = BIT(30)
};

enum {
	PXI_REGISTER_BASE = 0x0d800000
};

extern PPXI_REGISTERS HalpPxiRegisters;

#define PXI_REQUEST_READ() MmioReadBase32( MMIO_OFFSET(HalpPxiRegisters, Request.Message) )
#define PXI_REQUEST_WRITE(x) MmioWriteBase32( MMIO_OFFSET(HalpPxiRegisters, Request.Message), (ULONG)((x)) )
#define PXI_CONTROL_READ() ((PXI_CONTROL) MmioReadBase32( MMIO_OFFSET(HalpPxiRegisters, Request.Control) ))
#define PXI_CONTROL_WRITE(x) MmioWriteBase32( MMIO_OFFSET(HalpPxiRegisters, Request.Control), (ULONG)((x)) )
#define PXI_CONTROL_SET(x) do { \
	PXI_CONTROL Control = PXI_CONTROL_READ() & PXI_BITS_PRESERVE; \
	Control |= (x); \
	PXI_CONTROL_WRITE(Control); \
} while (FALSE)
#define PXI_RESPONSE_READ() MmioReadBase32( MMIO_OFFSET(HalpPxiRegisters, Response.Message) )
#define VEGAS_INTERRUPT_MASK_SET(x) MmioWriteBase32( MMIO_OFFSET(HalpPxiRegisters, InterruptMask), (ULONG)((x)) )
#define VEGAS_INTERRUPT_CLEAR(x) MmioWriteBase32( MMIO_OFFSET(HalpPxiRegisters, InterruptCause), (ULONG)((x)) )