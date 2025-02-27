// Flipper Command Processor.
#pragma once

enum {
	CP_FIFO_PHYS_ADDR = 0x0C008000,
	CP_FIFO_BAT_ADDR = 0x8C008000, // mapped by HAL to DBAT1
	HID2_WPE = 0x40000000,
	
	PE_BAT_ADDR = 0x8C001000,
	PI_BAT_ADDR = 0x8C003000,
	CP_BAT_ADDR = 0x8C000000
};

typedef struct _PE_REGISTERS {
	USHORT Unused[5];
	USHORT IntControl;
	USHORT IntStatus;
} PE_REGISTERS, *PPE_REGISTERS;

typedef struct _PI_REGISTERS {
	ULONG Unused[3];
	ULONG CpBase;
	ULONG CpTop;
	ULONG CpWrite;
} PI_REGISTERS, *PPI_REGISTERS;

typedef struct _CP_REGISTERS {
	USHORT Status;
	USHORT Enable;
	USHORT Clear;
	USHORT Unused[0x10 - 3];
	USHORT FifoBaseLow, FifoBaseHigh;
	USHORT FifoEndLow, FifoEndHigh;
	USHORT FifoHighMarkLow, FifoHighMarkHigh;
	USHORT FifoLowMarkLow, FifoLowMarkHigh;
	USHORT FifoCountLow, FifoCountHigh;
	USHORT FifoWritePointerLow, FifoWritePointerHigh;
	USHORT FifoReadPointerLow, FifoReadPointerHigh;
	USHORT FifoBreakPointLow, FifoBreakPointHigh;
} CP_REGISTERS, *PCP_REGISTERS;

#define BIT(x) (1 << (x))

enum {
	PIINT_PE_FINISHED = BIT(10),
	PIINT_CP_FIFO = BIT(11),
	PE_FINISHED_CLEAR = BIT(3),
	PICPWRITE_WRAPPED = BIT(29),
};

#define PE_REGISTER_BAT ((PPE_REGISTERS)PE_BAT_ADDR)
#define PI_REGISTER_BAT ((PPI_REGISTERS)PI_BAT_ADDR)
#define CP_REGISTER_BAT ((PCP_REGISTERS)CP_BAT_ADDR)

#define PE_FINISHED_RENDER (( MmioReadBase32( (PVOID)PI_BAT_ADDR, 0 ) & PIINT_PE_FINISHED ) != 0)
#define PE_READ_INT_CONTROL() MmioReadBase16( MMIO_OFFSET( PE_REGISTER_BAT, IntControl ) )
#define PE_FINISHED_CLEAR() MmioWriteBase16( MMIO_OFFSET( PE_REGISTER_BAT, IntControl ), PE_READ_INT_CONTROL() | PE_FINISHED_CLEAR )

#define PE_FIFO_INT() (( MmioReadBase32( (PVOID)PI_BAT_ADDR, 0 ) & PIINT_CP_FIFO ) != 0)
#define PE_FIFO_INTSTAT() MmioReadBase16( MMIO_OFFSET( CP_REGISTER_BAT, Status ) )
#define PE_FIFO_CLEAR() MmioWriteBase16( MMIO_OFFSET( CP_REGISTER_BAT, Clear ), 3 )

#define CP_READ32(Elem) ( ((ULONG)MmioReadBase16( MMIO_OFFSET( CP_REGISTER_BAT, Elem##High ) ) << 16) | MmioReadBase16( MMIO_OFFSET( CP_REGISTER_BAT, Elem##Low ) ) )

#define PI_READ_CP_WRITE() MmioReadBase32( MMIO_OFFSET( PI_REGISTER_BAT, CpWrite ) )
#define PI_CP_WRAPPED() ( ( PI_READ_CP_WRITE() & PICPWRITE_WRAPPED ) != 0 )
#define PI_CP_WRITE_ADDR() ( PI_READ_CP_WRITE() & (PICPWRITE_WRAPPED - 1) )

#define __mfspr(spr)    \
  ({ ULONG mfsprResult; \
     __asm__ volatile ("mfspr %0, %1" : "=r" (mfsprResult) : "n" (spr)); \
     mfsprResult; })

#define __mtspr(spr, value)     \
  __asm__ volatile ("mtspr %0, %1" : : "n" (spr), "r" (value))

#define SPR_WPAR 921
#define SPR_HID2 920

static inline BOOLEAN CppFifoNotEmpty(void) {
	return (__mfspr(SPR_WPAR) & 1);
}

static inline void CppFifoEnable(void) {
	// set the write pipe address register to physaddr of the CP fifo
	__mtspr(SPR_WPAR, CP_FIFO_PHYS_ADDR);
	// enable the write gather pipe
	__mtspr(SPR_HID2, __mfspr(SPR_HID2) | HID2_WPE);
}

static inline void CppWrite32(ULONG x)
{
	NativeWriteBase32((PVOID)CP_FIFO_BAT_ADDR, 0, x);
}

static inline void CppWrite16(USHORT x)
{
	NativeWriteBase16((PVOID)CP_FIFO_BAT_ADDR, 0, x);
}

static inline void CppWrite8(UCHAR x)
{
	NativeWriteBase8((PVOID)CP_FIFO_BAT_ADDR, 0, x);
}

static inline void CppWriteBpReg(ULONG x) {
	CppWrite8(0x61);
	CppWrite32(x);
}

// BP opcodes, shifted into the correct place.
enum
{
  BPMEM_GENMODE = 0x00 << 24,
  BPMEM_DISPLAYCOPYFILTER = 0x01 << 24,  // 0x01 + 4
  BPMEM_IND_MTXA = 0x06 << 24,           // 0x06 + (3 * 3)
  BPMEM_IND_MTXB = 0x07 << 24,           // 0x07 + (3 * 3)
  BPMEM_IND_MTXC = 0x08 << 24,           // 0x08 + (3 * 3)
  BPMEM_IND_IMASK = 0x0F << 24,
  BPMEM_IND_CMD = 0x10 << 24,  // 0x10 + 16
  BPMEM_SCISSORTL = 0x20 << 24,
  BPMEM_SCISSORBR = 0x21 << 24,
  BPMEM_LINEPTWIDTH = 0x22 << 24,
  BPMEM_PERF0_TRI = 0x23 << 24,
  BPMEM_PERF0_QUAD = 0x24 << 24,
  BPMEM_RAS1_SS0 = 0x25 << 24,
  BPMEM_RAS1_SS1 = 0x26 << 24,
  BPMEM_IREF = 0x27 << 24,
  BPMEM_TREF = 0x28 << 24,      // 0x28 + 8
  BPMEM_SU_SSIZE = 0x30 << 24,  // 0x30 + (2 * 8)
  BPMEM_SU_TSIZE = 0x31 << 24,  // 0x31 + (2 * 8)
  BPMEM_ZMODE = 0x40 << 24,
  BPMEM_BLENDMODE = 0x41 << 24,
  BPMEM_CONSTANTALPHA = 0x42 << 24,
  BPMEM_ZCOMPARE = 0x43 << 24,
  BPMEM_FIELDMASK = 0x44 << 24,
  BPMEM_SETDRAWDONE = 0x45 << 24,
  BPMEM_BUSCLOCK0 = 0x46 << 24,
  BPMEM_PE_TOKEN_ID = 0x47 << 24,
  BPMEM_PE_TOKEN_INT_ID = 0x48 << 24,
  BPMEM_EFB_TL = 0x49 << 24,
  BPMEM_EFB_WH = 0x4A << 24,
  BPMEM_EFB_ADDR = 0x4B << 24,
  BPMEM_EFB_STRIDE = 0x4D << 24,
  BPMEM_COPYYSCALE = 0x4E << 24,
  BPMEM_CLEAR_AR = 0x4F << 24,
  BPMEM_CLEAR_GB = 0x50 << 24,
  BPMEM_CLEAR_Z = 0x51 << 24,
  BPMEM_TRIGGER_EFB_COPY = 0x52 << 24,
  BPMEM_COPYFILTER0 = 0x53 << 24,
  BPMEM_COPYFILTER1 = 0x54 << 24,
  BPMEM_CLEARBBOX1 = 0x55 << 24,
  BPMEM_CLEARBBOX2 = 0x56 << 24,
  BPMEM_CLEAR_PIXEL_PERF = 0x57 << 24,
  BPMEM_REVBITS = 0x58 << 24,
  BPMEM_SCISSOROFFSET = 0x59 << 24,
  BPMEM_PRELOAD_ADDR = 0x60 << 24,
  BPMEM_PRELOAD_TMEMEVEN = 0x61 << 24,
  BPMEM_PRELOAD_TMEMODD = 0x62 << 24,
  BPMEM_PRELOAD_MODE = 0x63 << 24,
  BPMEM_LOADTLUT0 = 0x64 << 24,
  BPMEM_LOADTLUT1 = 0x65 << 24,
  BPMEM_TEXINVALIDATE = 0x66 << 24,
  BPMEM_PERF1 = 0x67 << 24,
  BPMEM_FIELDMODE = 0x68 << 24,
  BPMEM_BUSCLOCK1 = 0x69 << 24,
  BPMEM_TX_SETMODE0 = 0x80 << 24,     // 0x80 + 4
  BPMEM_TX_SETMODE1 = 0x84 << 24,     // 0x84 + 4
  BPMEM_TX_SETIMAGE0 = 0x88 << 24,    // 0x88 + 4
  BPMEM_TX_SETIMAGE1 = 0x8C << 24,    // 0x8C + 4
  BPMEM_TX_SETIMAGE2 = 0x90 << 24,    // 0x90 + 4
  BPMEM_TX_SETIMAGE3 = 0x94 << 24,    // 0x94 + 4
  BPMEM_TX_SETTLUT = 0x98 << 24,      // 0x98 + 4
  BPMEM_TX_SETMODE0_4 = 0xA0 << 24,   // 0xA0 + 4
  BPMEM_TX_SETMODE1_4 = 0xA4 << 24,   // 0xA4 + 4
  BPMEM_TX_SETIMAGE0_4 = 0xA8 << 24,  // 0xA8 + 4
  BPMEM_TX_SETIMAGE1_4 = 0xAC << 24,  // 0xA4 + 4
  BPMEM_TX_SETIMAGE2_4 = 0xB0 << 24,  // 0xB0 + 4
  BPMEM_TX_SETIMAGE3_4 = 0xB4 << 24,  // 0xB4 + 4
  BPMEM_TX_SETTLUT_4 = 0xB8 << 24,    // 0xB8 + 4
  BPMEM_TEV_COLOR_ENV = 0xC0 << 24,   // 0xC0 + (2 * 16)
  BPMEM_TEV_ALPHA_ENV = 0xC1 << 24,   // 0xC1 + (2 * 16)
  BPMEM_TEV_COLOR_RA = 0xE0 << 24,    // 0xE0 + (2 * 4)
  BPMEM_TEV_COLOR_BG = 0xE1 << 24,    // 0xE1 + (2 * 4)
  BPMEM_FOGRANGE = 0xE8 << 24,        // 0xE8 + 6
  BPMEM_FOGPARAM0 = 0xEE << 24,
  BPMEM_FOGBMAGNITUDE = 0xEF << 24,
  BPMEM_FOGBEXPONENT = 0xF0 << 24,
  BPMEM_FOGPARAM3 = 0xF1 << 24,
  BPMEM_FOGCOLOR = 0xF2 << 24,
  BPMEM_ALPHACOMPARE = 0xF3 << 24,
  BPMEM_BIAS = 0xF4 << 24,
  BPMEM_ZTEX2 = 0xF5 << 24,
  BPMEM_TEV_KSEL = 0xF6 << 24,  // 0xF6 + 8
  BPMEM_BP_MASK = 0xFE << 24,
};