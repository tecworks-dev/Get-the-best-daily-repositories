// Defines runtime variables, shared from ARC fw to OS.
#pragma once

// if we're building for NT, then we're LE.
#ifndef ARC_LE
#define __BUILDING_FOR_NT__
#define ARC_LE
#define ARC_BE __attribute__((scalar_storage_order("big-endian")))
#define ARC_ALIGNED(x) __attribute__((aligned(x)))
#define ARC_PACKED __attribute__((packed))
#define BIT(x) (1 << (x))

#define RUNTIME_BLOCK (*(PVOID**)(0x8000403C))

typedef struct ARC_BE _U32BE {
	ULONG v;
} U32BE, *PU32BE;

#endif

#define _MMIO_SIZEOF(Pointer, Element) sizeof((Pointer)->Element)
#define _MMIO_OFFSETOF(Pointer, Element) ((ULONG)(&(Pointer)->Element) - (ULONG)(Pointer))
#define MMIO_OFFSET(Pointer, Element) Pointer, _MMIO_OFFSETOF(Pointer, Element)

static inline void _Mmio_Barrier(UCHAR Write) {
	if (Write) __asm__ __volatile__ ("eieio");
	else __asm__ __volatile__ ("sync");
}

static inline BOOLEAN _Mmio_IsLittleEndian(void) {
	// Flipper and derivatives are always big endian.
	// (besides some blocks in Vegas that endianness swap incorrectly)
	// (and Latte has a Radeon which like all Radeons can be either endianness)
	return 0;
}

// Given the address, and the length to access, munge the address to counteract what the CPU does with MSR_LE enabled.
// Length must be 1, 2, 4, or 8, this won't be checked for.
static inline ULONG _Mmio_MungeAddressConstant(ULONG Length) {
	// 1 => 7, 2 => 6, 4 => 4, 8 => 0
	// this is enough, and should be calculated at compile time :)
	return (8 - Length);
}
static inline PVOID _Mmio_MungeAddressForBig(PVOID Address, ULONG Length) {
	// do nothing for 64 bits.
	if (Length == 8) return Address;
	
	ULONG Addr32 = (ULONG)Address;
	ULONG AlignOffset = Addr32 & (Length - 1);
	if (AlignOffset == 0) {
		// Aligned access, just XOR with munge constant.
		if (__builtin_constant_p(Addr32) && Length == 4) {
			// Optimise for the case of constant address and 32-bit access:
			if ((Addr32 & 4) != 0) return (PVOID)(Addr32 - 4);
			return (PVOID)(Addr32 + 4);
		}
		return (PVOID)(Addr32 ^ _Mmio_MungeAddressConstant(Length));
	}
	
	// Unaligned access.
	// Convert the address to an aligned address.
	Addr32 &= ~(Length - 1);
	// XOR with munge constant
	Addr32 ^= _Mmio_MungeAddressConstant(Length);
	// And subtract the align offset.
	return (PVOID)(Addr32 - AlignOffset);
}

// Used for the variations which take base address + constant offset (base address must be 64-bit aligned!)
static inline ULONG _Mmio_MungeOffsetForBig(ULONG Offset, ULONG Length) {
	// do nothing for 64 bits.
	if (Length == 8) return Offset;

	ULONG AlignOffset = Offset & (Length - 1);
	if (AlignOffset == 0) {
		// Aligned access, just XOR with munge constant.
		return (Offset ^ _Mmio_MungeAddressConstant(Length));
	}

	// Unaligned access.
	// Convert the address to an aligned address.
	Offset &= ~(Length - 1);
	// XOR with munge constant.
	Offset ^= _Mmio_MungeAddressConstant(Length);
	// And subtract the align offset.
	return Offset - AlignOffset;
}

static inline UCHAR _Mmio_Read8Impl(PVOID base, ULONG offset, BOOLEAN little, BOOLEAN barrier) {
	(void)little;
	// inline asm here generates the smallest code-size for multiple accesses to same register space
	//UCHAR x = *(volatile UCHAR*)((ULONG)base + offset);
	UCHAR x;
	if (__builtin_constant_p(offset) && offset < 0x8000) __asm__ __volatile__("lbz %0, %1(%2)" : "=r"(x) : "I"(offset), "b"(base) : "memory");
	else __asm__ __volatile__("lbzx %0, %2, %1" : "=r"(x) : "r"(base), "b"(offset) : "memory");
	if (barrier) _Mmio_Barrier(0);
	return x;
}

static inline void _Mmio_Write8Impl(PVOID base, ULONG offset, UCHAR value, BOOLEAN little, BOOLEAN barrier) {
	(void)little;
	// inline asm here generates the smallest code-size for multiple accesses to same register space
	//*(volatile UCHAR*)((ULONG)base + offset) = value;
	if (__builtin_constant_p(offset) && offset < 0x8000) __asm__ __volatile__("stb %0, %1(%2)"  : : "r"(value), "I"(offset), "b"(base) : "memory");
	else __asm__ __volatile__("stbx %0, %2, %1"  : : "r"(value), "r"(base), "b"(offset) : "memory");
	if (barrier) _Mmio_Barrier(1);
}

static inline USHORT _Mmio_Read16Impl(PVOID base, ULONG offset, BOOLEAN little, BOOLEAN barrier) {
	USHORT x;
	if (little) {
		// inline asm here generates the smallest code-size for multiple accesses to same register space
		// check for constant zero offset, to avoid wasting a register there
		//x = __builtin_bswap16(*(volatile USHORT*)((ULONG)base + offset));
		if (__builtin_constant_p(offset) && offset == 0) __asm__ __volatile__("lhbrx %0, 0, %1" : "=r"(x) : "r"(base) : "memory");
		else __asm__ __volatile__("lhbrx %0, %2, %1" : "=r"(x) : "r"(base), "b"(offset) : "memory");
		if (barrier) _Mmio_Barrier(0);
		return x;
	}
	else {
		// inline asm here generates the smallest code-size for multiple accesses to same register space
		//x = *(volatile USHORT*)((ULONG)base + offset);
		if (__builtin_constant_p(offset) && offset < 0x8000) __asm__ __volatile__("lhz %0, %1(%2)" : "=r"(x) : "I"(offset), "b"(base) : "memory");
		else __asm__ __volatile__("lhzx %0, %2, %1" : "=r"(x) : "r"(base), "b"(offset) : "memory");
		if (barrier) _Mmio_Barrier(0);
		return x;
	}
}

static inline void _Mmio_Write16Impl(PVOID base, ULONG offset, USHORT value, BOOLEAN little, BOOLEAN barrier) {
	if (little) {
		// inline asm here generates the smallest code-size for multiple accesses to same register space
		// check for constant zero offset, to avoid wasting a register there
		//*(volatile USHORT*)((ULONG)base + offset) = __builtin_bswap16(value);
		if (__builtin_constant_p(offset) && offset == 0) __asm__ __volatile__("sthbrx %0, 0, %1"  : : "r"(value), "r"(base) : "memory");
		else __asm__ __volatile__("sthbrx %0, %2, %1"  : : "r"(value), "r"(base), "b"(offset) : "memory");
		if (barrier) _Mmio_Barrier(1);
		return;
	}
	else {
		// inline asm here generates the smallest code-size for multiple accesses to same register space
		//*(volatile USHORT*)((ULONG)base + offset) = value;
		if (__builtin_constant_p(offset) && offset < 0x8000) __asm__ __volatile__("sth %0, %1(%2)"  : : "r"(value), "I"(offset), "b"(base) : "memory");
		else __asm__ __volatile__("sthx %0, %2, %1"  : : "r"(value), "r"(base), "b"(offset) : "memory");
		if (barrier) _Mmio_Barrier(1);
		return;
	}
}

static inline ULONG _Mmio_Read32Impl(PVOID base, ULONG offset, BOOLEAN little, BOOLEAN barrier) {
	ULONG x;
	if (little) {
		// inline asm here generates the smallest code-size for multiple accesses to same register space
		// check for constant zero offset, to avoid wasting a register there
		//x = __builtin_bswap32(*(volatile ULONG*)((ULONG)base + offset));
		if (__builtin_constant_p(offset) && offset == 0) __asm__ __volatile__("lwbrx %0, 0, %1" : "=r"(x) : "r"(base) : "memory");
		else __asm__ __volatile__("lwbrx %0, %2, %1" : "=r"(x) : "r"(base), "b"(offset) : "memory");
		if (barrier) _Mmio_Barrier(0);
		return x;
	}
	else {
		// inline asm here generates the smallest code-size for multiple accesses to same register space
		//x = *(volatile ULONG*)((ULONG)base + offset);
		if (__builtin_constant_p(offset) && offset < 0x8000) __asm__ __volatile__("lwz %0, %1(%2)" : "=r"(x) : "I"(offset), "b"(base) : "memory");
		else __asm__ __volatile__("lwzx %0, %2, %1" : "=r"(x) : "r"(base), "b"(offset) : "memory");
		if (barrier) _Mmio_Barrier(0);
		return x;
	}
}

static inline void _Mmio_Write32Impl(PVOID base, ULONG offset, ULONG value, BOOLEAN little, BOOLEAN barrier) {
	if (little) {
		// inline asm here generates the smallest code-size for multiple accesses to same register space
		// check for constant zero offset, to avoid wasting a register there
		//*(volatile ULONG*)((ULONG)base + offset) = __builtin_bswap32(value);
		if (__builtin_constant_p(offset) && offset == 0) __asm__ __volatile__("stwbrx %0, 0, %1"  : : "r"(value), "r"(base) : "memory");
		else __asm__ __volatile__("stwbrx %0, %2, %1"  : : "r"(value), "r"(base), "b"(offset) : "memory");
		if (barrier) _Mmio_Barrier(1);
		return;
	}
	else {
		// inline asm here generates the smallest code-size for multiple accesses to same register space
		//*(volatile ULONG*)((ULONG)base + offset) = value;
		if (__builtin_constant_p(offset) && offset < 0x8000) __asm__ __volatile__("stw %0, %1(%2)"  : : "r"(value), "I"(offset), "b"(base) : "memory");
		else __asm__ __volatile__("stwx %0, %2, %1"  : : "r"(value), "r"(base), "b"(offset) : "memory");
		if (barrier) _Mmio_Barrier(1);
		return;
	}
}

static inline UCHAR MmioRead8(PVOID addr) {
	if (!_Mmio_IsLittleEndian()) addr = _Mmio_MungeAddressForBig(addr, 1);
	return _Mmio_Read8Impl(addr, 0, 0, 1);
}

static inline void MmioWrite8(PVOID addr, UCHAR x) {
	if (!_Mmio_IsLittleEndian()) addr = _Mmio_MungeAddressForBig(addr, 1);
	_Mmio_Write8Impl(addr, 0, x, 0, 1);
}

static inline USHORT MmioRead16(PVOID addr)
{
	if (_Mmio_IsLittleEndian()) return _Mmio_Read16Impl(addr, 0, 1, 1);
	addr = _Mmio_MungeAddressForBig(addr, 2);
	return _Mmio_Read16Impl(addr, 0, 0, 1);
}

static inline void MmioWrite16(PVOID addr, USHORT x)
{
	if (_Mmio_IsLittleEndian()) {
		_Mmio_Write16Impl(addr, 0, x, 1, 1);
		return;
	}
	addr = _Mmio_MungeAddressForBig(addr, 2);
	_Mmio_Write16Impl(addr, 0, x, 0, 1);
}

static inline USHORT MmioRead16L(PVOID addr)
{
	if (_Mmio_IsLittleEndian()) return _Mmio_Read16Impl(addr, 0, 0, 1);
	addr = _Mmio_MungeAddressForBig(addr, 2);
	return _Mmio_Read16Impl(addr, 0, 1, 1);
}

static inline void MmioWrite16L(PVOID addr, USHORT x)
{
	if (_Mmio_IsLittleEndian()) {
		_Mmio_Write16Impl(addr, 0, x, 0, 1);
		return;
	}
	addr = _Mmio_MungeAddressForBig(addr, 2);
	_Mmio_Write16Impl(addr, 0, x, 1, 1);
}

static inline ULONG MmioRead32(PVOID addr)
{
	if (_Mmio_IsLittleEndian()) return _Mmio_Read32Impl(addr, 0, 1, 1);
	addr = _Mmio_MungeAddressForBig(addr, 4);
	return _Mmio_Read32Impl(addr, 0, 0, 1);
}

static inline void MmioWrite32(PVOID addr, ULONG x)
{
	if (_Mmio_IsLittleEndian()) {
		_Mmio_Write32Impl(addr, 0, x, 1, 1);
		return;
	}
	addr = _Mmio_MungeAddressForBig(addr, 4);
	return _Mmio_Write32Impl(addr, 0, x, 0, 1);
}

static inline ULONG MmioRead32L(PVOID addr)
{
	if (_Mmio_IsLittleEndian()) return _Mmio_Read32Impl(addr, 0, 0, 1);
	addr = _Mmio_MungeAddressForBig(addr, 4);
	return _Mmio_Read32Impl(addr, 0, 1, 1);
}

static inline void MmioWrite32L(PVOID addr, ULONG x)
{
	if (_Mmio_IsLittleEndian()) {
		_Mmio_Write32Impl(addr, 0, x, 0, 1);
		return;
	}
	addr = _Mmio_MungeAddressForBig(addr, 4);
	_Mmio_Write32Impl(addr, 0, x, 1, 1);
}

static inline UCHAR MmioReadBase8(PVOID base, ULONG offset) {
	if (!_Mmio_IsLittleEndian()) offset = _Mmio_MungeOffsetForBig(offset, 1);
	return _Mmio_Read8Impl(base, offset, 0, 1);
}

static inline void MmioWriteBase8(PVOID base, ULONG offset, UCHAR x) {
	if (!_Mmio_IsLittleEndian()) offset = _Mmio_MungeOffsetForBig(offset, 1);
	_Mmio_Write8Impl(base, offset, x, 0, 1);
}

static inline USHORT MmioReadBase16(PVOID base, ULONG offset) {
	if (_Mmio_IsLittleEndian()) return _Mmio_Read16Impl(base, offset, 1, 1);
	offset = _Mmio_MungeOffsetForBig(offset, 2);
	return _Mmio_Read16Impl(base, offset, 0, 1);
}

static inline void MmioWriteBase16(PVOID base, ULONG offset, USHORT x) {
	if (_Mmio_IsLittleEndian()) {
		_Mmio_Write16Impl(base, offset, x, 1, 1);
		return;
	}
	offset = _Mmio_MungeOffsetForBig(offset, 2);
	_Mmio_Write16Impl(base, offset, x, 0, 1);
}

static inline USHORT MmioReadBase16L(PVOID base, ULONG offset) {
	if (_Mmio_IsLittleEndian()) return _Mmio_Read16Impl(base, offset, 0, 1);
	offset = _Mmio_MungeOffsetForBig(offset, 2);
	return _Mmio_Read16Impl(base, offset, 1, 1);
}

static inline void MmioWriteBase16L(PVOID base, ULONG offset, USHORT x) {
	if (_Mmio_IsLittleEndian()) {
		_Mmio_Write16Impl(base, offset, x, 0, 1);
		return;
	}
	offset = _Mmio_MungeOffsetForBig(offset, 2);
	_Mmio_Write16Impl(base, offset, x, 1, 1);
}

static inline ULONG MmioReadBase32(PVOID base, ULONG offset) {
	if (_Mmio_IsLittleEndian()) return _Mmio_Read32Impl(base, offset, 1, 1);
	offset = _Mmio_MungeOffsetForBig(offset, 4);
	return _Mmio_Read32Impl(base, offset, 0, 1);
}

static inline void MmioWriteBase32(PVOID base, ULONG offset, ULONG x) {
	if (_Mmio_IsLittleEndian()) {
		_Mmio_Write32Impl(base, offset, x, 1, 1);
		return;
	}
	offset = _Mmio_MungeOffsetForBig(offset, 4);
	return _Mmio_Write32Impl(base, offset, x, 0, 1);
}

static inline ULONG MmioReadBase32L(PVOID base, ULONG offset) {
	if (_Mmio_IsLittleEndian()) return _Mmio_Read32Impl(base, offset, 0, 1);
	offset = _Mmio_MungeOffsetForBig(offset, 4);
	return _Mmio_Read32Impl(base, offset, 1, 1);
}

static inline void MmioWriteBase32L(PVOID base, ULONG offset, ULONG x) {
	if (_Mmio_IsLittleEndian()) {
		_Mmio_Write32Impl(base, offset, x, 0, 1);
		return;
	}
	offset = _Mmio_MungeOffsetForBig(offset, 4);
	_Mmio_Write32Impl(base, offset, x, 1, 1);
}

#define __MMIO_BUF_READ_BODY(func, little) \
	BOOLEAN sysLittle = _Mmio_IsLittleEndian(); \
	if (!sysLittle) addr = _Mmio_MungeAddressForBig(addr, sizeof(buf[0])); \
	sysLittle ^= little; \
	for (ULONG readCount = 0; readCount < len; readCount++) { \
		buf[readCount] = func(addr, 0, sysLittle, 1); \
	}

#define __MMIO_BUF_WRITE_BODY(func, little) \
	BOOLEAN sysLittle = _Mmio_IsLittleEndian(); \
	if (!sysLittle) addr = _Mmio_MungeAddressForBig(addr, sizeof(buf[0])); \
	sysLittle ^= little; \
	for (ULONG writeCount = 0; writeCount < len; writeCount++) { \
		func(addr, 0, buf[writeCount], sysLittle, 1); \
	}

#define __MMIO_BASEBUF_READ_BODY(func, little) \
	BOOLEAN sysLittle = _Mmio_IsLittleEndian(); \
	if (!sysLittle) offset = _Mmio_MungeOffsetForBig(offset, sizeof(buf[0])); \
	sysLittle ^= little; \
	for (ULONG readCount = 0; readCount < len; readCount++) { \
		buf[readCount] = func(base, offset, sysLittle, 1); \
	}

#define __MMIO_BASEBUF_WRITE_BODY(func, little) \
	BOOLEAN sysLittle = _Mmio_IsLittleEndian(); \
	if (!sysLittle) offset = _Mmio_MungeOffsetForBig(offset, sizeof(buf[0])); \
	sysLittle ^= little; \
	for (ULONG writeCount = 0; writeCount < len; writeCount++) { \
		func(base, offset, buf[writeCount], sysLittle, 1); \
	}

static inline void MmioReadBuf8(PVOID addr, PUCHAR buf, ULONG len) {
	__MMIO_BUF_READ_BODY(_Mmio_Read8Impl, 0);
}

static inline void MmioReadBuf16(PVOID addr, PUSHORT buf, ULONG len) {
	__MMIO_BUF_READ_BODY(_Mmio_Read16Impl, 0);
}

static inline void MmioReadBuf16L(PVOID addr, PUSHORT buf, ULONG len) {
	__MMIO_BUF_READ_BODY(_Mmio_Read16Impl, 1);
}

static inline void MmioReadBuf32(PVOID addr, PULONG buf, ULONG len) {
	__MMIO_BUF_READ_BODY(_Mmio_Read32Impl, 0);
}

static inline void MmioReadBuf32L(PVOID addr, PULONG buf, ULONG len) {
	__MMIO_BUF_READ_BODY(_Mmio_Read32Impl, 1);
}

static inline void MmioWriteBuf8(PVOID addr, PUCHAR buf, ULONG len) {
	__MMIO_BUF_WRITE_BODY(_Mmio_Write8Impl, 0);
}

static inline void MmioWriteBuf16(PVOID addr, PUSHORT buf, ULONG len) {
	__MMIO_BUF_WRITE_BODY(_Mmio_Write16Impl, 0);
}

static inline void MmioWriteBuf16L(PVOID addr, PUSHORT buf, ULONG len) {
	__MMIO_BUF_WRITE_BODY(_Mmio_Write16Impl, 1);
}

static inline void MmioWriteBuf32(PVOID addr, PULONG buf, ULONG len) {
	__MMIO_BUF_WRITE_BODY(_Mmio_Write32Impl, 0);
}

static inline void MmioWriteBuf32L(PVOID addr, PULONG buf, ULONG len) {
	__MMIO_BUF_WRITE_BODY(_Mmio_Write32Impl, 1);
}

static inline void MmioReadBaseBuf8(PVOID base, ULONG offset, PUCHAR buf, ULONG len) {
	__MMIO_BASEBUF_READ_BODY(_Mmio_Read8Impl, 0);
}

static inline void MmioReadBaseBuf16(PVOID base, ULONG offset, PUSHORT buf, ULONG len) {
	__MMIO_BASEBUF_READ_BODY(_Mmio_Read16Impl, 0);
}

static inline void MmioReadBaseBuf16L(PVOID base, ULONG offset, PUSHORT buf, ULONG len) {
	__MMIO_BASEBUF_READ_BODY(_Mmio_Read16Impl, 1);
}

static inline void MmioReadBaseBuf32(PVOID base, ULONG offset, PULONG buf, ULONG len) {
	__MMIO_BASEBUF_READ_BODY(_Mmio_Read32Impl, 0);
}

static inline void MmioReadBaseBuf32L(PVOID base, ULONG offset, PULONG buf, ULONG len) {
	__MMIO_BASEBUF_READ_BODY(_Mmio_Read32Impl, 1);
}

static inline void MmioWriteBaseBuf8(PVOID base, ULONG offset, PUCHAR buf, ULONG len) {
	__MMIO_BASEBUF_WRITE_BODY(_Mmio_Write8Impl, 0);
}

static inline void MmioWriteBaseBuf16(PVOID base, ULONG offset, PUSHORT buf, ULONG len) {
	__MMIO_BASEBUF_WRITE_BODY(_Mmio_Write16Impl, 0);
}

static inline void MmioWriteBaseBuf16L(PVOID base, ULONG offset, PUSHORT buf, ULONG len) {
	__MMIO_BASEBUF_WRITE_BODY(_Mmio_Write16Impl, 1);
}

static inline void MmioWriteBaseBuf32(PVOID base, ULONG offset, PULONG buf, ULONG len) {
	__MMIO_BASEBUF_WRITE_BODY(_Mmio_Write32Impl, 0);
}

static inline void MmioWriteBaseBuf32L(PVOID base, ULONG offset, PULONG buf, ULONG len) {
	__MMIO_BASEBUF_WRITE_BODY(_Mmio_Write32Impl, 1);
}

static inline ULONG LoadToRegister32(ULONG value) {
	asm volatile ("" : : "r"(value));
	return value;
}

static inline ULONG EfbRead32(PVOID addr)
{
	ULONG x;
	addr = _Mmio_MungeAddressForBig(addr, 4);
	__asm__ __volatile__(
		"sync ; lwz %0,0(%1)" : "=r"(x) : "b"(addr));
	return x;
}

static inline void EfbWrite32(PVOID addr, ULONG x)
{
	addr = _Mmio_MungeAddressForBig(addr, 4);
	__asm__ __volatile__(
		"sync ; stw %0,0(%1)" : : "r"(x), "b"(addr));
	_Mmio_Barrier(1);
}

static inline ULONG NativeRead32(PVOID addr)
{
	if (!_Mmio_IsLittleEndian()) addr = _Mmio_MungeAddressForBig(addr, 4);
	return _Mmio_Read32Impl(addr, 0, 0, 0);
}

static inline void NativeWrite32(PVOID addr, ULONG x)
{
	if (!_Mmio_IsLittleEndian()) addr = _Mmio_MungeAddressForBig(addr, 4);
	_Mmio_Write32Impl(addr, 0, x, 0, 0);
}

static inline USHORT NativeRead16(PVOID addr)
{
	if (!_Mmio_IsLittleEndian()) addr = _Mmio_MungeAddressForBig(addr, 2);
	return _Mmio_Read16Impl(addr, 0, 0, 0);
}

static inline void NativeWrite16(PVOID addr, USHORT x)
{
	if (!_Mmio_IsLittleEndian()) addr = _Mmio_MungeAddressForBig(addr, 2);
	_Mmio_Write16Impl(addr, 0, x, 0, 0);
}

static inline USHORT NativeRead8(PVOID addr)
{
	if (!_Mmio_IsLittleEndian()) addr = _Mmio_MungeAddressForBig(addr, 1);
	return _Mmio_Read8Impl(addr, 0, 0, 0);
}

static inline void NativeWrite8(PVOID addr, UCHAR x)
{
	if (!_Mmio_IsLittleEndian()) addr = _Mmio_MungeAddressForBig(addr, 1);
	_Mmio_Write8Impl(addr, 0, x, 0, 0);
}

static inline ULONG NativeReadBase32(PVOID addr, ULONG offset)
{
	if (!_Mmio_IsLittleEndian()) offset = _Mmio_MungeOffsetForBig(offset, 4);
	return _Mmio_Read32Impl(addr, offset, 0, 0);
}

static inline void NativeWriteBase32(PVOID addr, ULONG offset, ULONG x)
{
	if (!_Mmio_IsLittleEndian()) offset = _Mmio_MungeOffsetForBig(offset, 4);
	_Mmio_Write32Impl(addr, offset, x, 0, 0);
}

static inline USHORT NativeReadBase16(PVOID addr, ULONG offset)
{
	if (!_Mmio_IsLittleEndian()) offset = _Mmio_MungeOffsetForBig(offset, 2);
	return _Mmio_Read16Impl(addr, offset, 0, 0);
}

static inline void NativeWriteBase16(PVOID addr, ULONG offset, USHORT x)
{
	if (!_Mmio_IsLittleEndian()) offset = _Mmio_MungeOffsetForBig(offset, 2);
	_Mmio_Write16Impl(addr, offset, x, 0, 0);
}

static inline USHORT NativeReadBase8(PVOID addr, ULONG offset)
{
	if (!_Mmio_IsLittleEndian()) offset = _Mmio_MungeOffsetForBig(offset, 1);
	return _Mmio_Read8Impl(addr, offset, 0, 0);
}

static inline void NativeWriteBase8(PVOID addr, ULONG offset, UCHAR x)
{
	if (!_Mmio_IsLittleEndian()) offset = _Mmio_MungeOffsetForBig(offset, 1);
	_Mmio_Write8Impl(addr, offset, x, 0, 0);
}

// Write a value to a structure in the correct address for some other hardware to be able to correctly access it.
#define NATIVE_WRITE(ptr, element, value) do { \
	switch (_MMIO_SIZEOF(ptr, element)) { \
	case 1: NativeWriteBase8(MMIO_OFFSET(ptr, element), value); break; \
	case 2: NativeWriteBase16(MMIO_OFFSET(ptr, element), value); break; \
	case 4: NativeWriteBase32(MMIO_OFFSET(ptr, element), value); break; \
	default: { _Static_assert(_MMIO_SIZEOF(ptr, element) == 1 || _MMIO_SIZEOF(ptr, element) == 2 || _MMIO_SIZEOF(ptr, element) == 4); } break; \
	} \
} while (0)

#define NATIVE_COPY_FROM(dest, src, element) do {\	switch (_MMIO_SIZEOF(src, element)) { \
	case 1: dest -> element = NativeReadBase8(MMIO_OFFSET(src, element), value); break; \
	case 2: dest -> element = NativeReadBase16(MMIO_OFFSET(src, element), value); break; \
	case 4: dest -> element = NativeReadBase32(MMIO_OFFSET(src, element), value); break; \
	default: { _Static_assert(_MMIO_SIZEOF(src, element) == 1 || _MMIO_SIZEOF(src, element) == 2 || _MMIO_SIZEOF(src, element) == 4); } break; \
	} \
} while (0)

#define NATIVE_COPY_FROM_OFFSET(dest, src, offset, element) do {\	switch (_MMIO_SIZEOF(dest, element)) { \
	case 1: dest -> element = NativeReadBase8((PVOID)((ULONG)src - offset), _MMIO_OFFSETOF(src, element) + offset, value); break; \
	case 2: dest -> element = NativeReadBase16((PVOID)((ULONG)src - offset), _MMIO_OFFSETOF(src, element) + offset, value); break; \
	case 4: dest -> element = NativeReadBase32((PVOID)((ULONG)src - offset), _MMIO_OFFSETOF(src, element) + offset, value); break; \
	default: { _Static_assert(_MMIO_SIZEOF(src, element) == 1 || _MMIO_SIZEOF(src, element) == 2 || _MMIO_SIZEOF(src, element) == 4); } break; \
	} \
} while (0)

#define NATIVE_COPY_FROM_OFFSET_OFFSET(dest, src, offset, elementDest, elementSrc, element) do {\	switch (_MMIO_SIZEOF(dest, elementDest . element)) { \
	case 1: dest -> elementDest . element = NativeReadBase8((PVOID)((ULONG)src - offset), _MMIO_OFFSETOF(src, elementSrc . element) + offset, value); break; \
	case 2: dest -> elementDest . element = NativeReadBase16((PVOID)((ULONG)src - offset), _MMIO_OFFSETOF(src, elementSrc . element) + offset, value); break; \
	case 4: dest -> elementDest . element = NativeReadBase32((PVOID)((ULONG)src - offset), _MMIO_OFFSETOF(src, elementSrc . element) + offset, value); break; \
	default: { _Static_assert(_MMIO_SIZEOF(src, elementSrc . element) == 1 || _MMIO_SIZEOF(src, elementSrc . element) == 2 || _MMIO_SIZEOF(src, elementSrc . element) == 4); } break; \
	} \
} while (0)

#define NATIVE_COPY_TO(dest, src, element) NATIVE_WRITE(dest, element, src -> element)

// Array initialiser for a byteswapped string.
#define __STRING_BYTESWAP_ELEM(str, i) (i^7) >= sizeof(str) ? 0 : str[i^7]
#define STRING_BYTESWAP(str) { \
	__STRING_BYTESWAP_ELEM(str, 0), __STRING_BYTESWAP_ELEM(str, 1), __STRING_BYTESWAP_ELEM(str, 2), __STRING_BYTESWAP_ELEM(str, 3), \
	__STRING_BYTESWAP_ELEM(str, 4), __STRING_BYTESWAP_ELEM(str, 5), __STRING_BYTESWAP_ELEM(str, 6), __STRING_BYTESWAP_ELEM(str, 7), \
	__STRING_BYTESWAP_ELEM(str, 8), __STRING_BYTESWAP_ELEM(str, 9), __STRING_BYTESWAP_ELEM(str, 10), __STRING_BYTESWAP_ELEM(str, 11), \
	__STRING_BYTESWAP_ELEM(str, 12), __STRING_BYTESWAP_ELEM(str, 13), __STRING_BYTESWAP_ELEM(str, 14), __STRING_BYTESWAP_ELEM(str, 15), \
	__STRING_BYTESWAP_ELEM(str, 16), __STRING_BYTESWAP_ELEM(str, 17), __STRING_BYTESWAP_ELEM(str, 18), __STRING_BYTESWAP_ELEM(str, 19), \
	__STRING_BYTESWAP_ELEM(str, 20), __STRING_BYTESWAP_ELEM(str, 21), __STRING_BYTESWAP_ELEM(str, 22), __STRING_BYTESWAP_ELEM(str, 23), \
	__STRING_BYTESWAP_ELEM(str, 24), __STRING_BYTESWAP_ELEM(str, 25), __STRING_BYTESWAP_ELEM(str, 26), __STRING_BYTESWAP_ELEM(str, 27), \
	__STRING_BYTESWAP_ELEM(str, 28), __STRING_BYTESWAP_ELEM(str, 29), __STRING_BYTESWAP_ELEM(str, 30), __STRING_BYTESWAP_ELEM(str, 31), \
	__STRING_BYTESWAP_ELEM(str, 32), __STRING_BYTESWAP_ELEM(str, 33), __STRING_BYTESWAP_ELEM(str, 34), __STRING_BYTESWAP_ELEM(str, 35), \
	__STRING_BYTESWAP_ELEM(str, 36), __STRING_BYTESWAP_ELEM(str, 37), __STRING_BYTESWAP_ELEM(str, 38), __STRING_BYTESWAP_ELEM(str, 39), \
	__STRING_BYTESWAP_ELEM(str, 40), __STRING_BYTESWAP_ELEM(str, 41), __STRING_BYTESWAP_ELEM(str, 42), __STRING_BYTESWAP_ELEM(str, 43), \
	__STRING_BYTESWAP_ELEM(str, 44), __STRING_BYTESWAP_ELEM(str, 45), __STRING_BYTESWAP_ELEM(str, 46), __STRING_BYTESWAP_ELEM(str, 47), \
	__STRING_BYTESWAP_ELEM(str, 48), __STRING_BYTESWAP_ELEM(str, 49), __STRING_BYTESWAP_ELEM(str, 50), __STRING_BYTESWAP_ELEM(str, 51), \
	__STRING_BYTESWAP_ELEM(str, 52), __STRING_BYTESWAP_ELEM(str, 53), __STRING_BYTESWAP_ELEM(str, 54), __STRING_BYTESWAP_ELEM(str, 55), \
	__STRING_BYTESWAP_ELEM(str, 56), __STRING_BYTESWAP_ELEM(str, 57), __STRING_BYTESWAP_ELEM(str, 58), __STRING_BYTESWAP_ELEM(str, 59), \
	__STRING_BYTESWAP_ELEM(str, 60), __STRING_BYTESWAP_ELEM(str, 61), __STRING_BYTESWAP_ELEM(str, 62), __STRING_BYTESWAP_ELEM(str, 63), \
}

//#ifndef RUNTIME_NO_ARC
//#include <arc.h>
//#define RUNTIME_BLOCK (*(PVOID**)((ULONG)SYSTEM_BLOCK + sizeof(SYSTEM_PARAMETER_BLOCK)))
//#endif

typedef enum {
	/// <summary>
	/// Nintendo GameCube / "Dolphin"/DOL - Gekko + Flipper, 24MB or 48MB of Splash/MEM1. 
	/// A theoretically supported early Broadway Evaluation Board (Broadway + Flipper, 48MB of Splash/MEM1) also counts here.
	/// </summary>
	ARTX_SYSTEM_FLIPPER,
	/// <summary>
	/// Nintendo Wii / "Revolution"/RVL - Broadway + Vegas/Bollywood, 24MB of Napa/MEM1 + 64MB or 128MB of DDR/MEM2. 
	/// A theoretically supported early Cortado board (Broadway + Vegas/Bollywood, Espresso + Bollywood, unknown DDR sizes) also counts here.
	/// </summary>
	ARTX_SYSTEM_VEGAS,
	/// <summary>
	/// Nintendo Wii U / "Cafe"/WUP - Espresso + Latte, 32MB of MEM1 + 2GB or 3GB of DDR/MEM2 (+ EFB usable as RAM)
	/// </summary>
	ARTX_SYSTEM_LATTE
} ARTX_SYSTEM_TYPE;

enum {
	RUNTIME_FRAME_BUFFER,
	RUNTIME_DECREMENTER_FREQUENCY,
	RUNTIME_RTC_BIAS,
	RUNTIME_SYSTEM_TYPE,
	RUNTIME_IPC_AREA,
	RUNTIME_IN_EMULATOR,
	RUNTIME_GX_FIFO,
	RUNTIME_ENV_DISK,
	RUNTIME_RESET_STUB,
	RUNTIME_EXI_DEVICES,
	RUNTIME_USB_DEVICES
};

typedef struct ARC_LE _FRAME_BUFFER {
	union ARC_LE {
		PVOID Pointer;
		ULONG PointerArc;
	};
	ULONG Length;
	ULONG Width;
	ULONG Height;
	ULONG Stride;
} FRAME_BUFFER, *PFRAME_BUFFER;

typedef struct ARC_LE _MEMORY_AREA {
	ULONG PointerArc;
	ULONG Length;
} MEMORY_AREA, *PMEMORY_AREA;

#ifndef __BUILDING_FOR_NT__
typedef struct ARC_LE {
	U32LE RuntimePointers[16];
	FRAME_BUFFER RuntimeFb;
	MEMORY_AREA IpcArea;
	MEMORY_AREA GxFifo;
} RUNTIME_AREA, * PRUNTIME_AREA;

#define s_RuntimeArea ((PRUNTIME_AREA)0x80005000)
#define s_RuntimePointers s_RuntimeArea->RuntimePointers
#define s_RuntimeFb s_RuntimeArea->RuntimeFb
#define s_RuntimeIpc s_RuntimeArea->IpcArea
#define s_RuntimeGx s_RuntimeArea->GxFifo

_Static_assert(sizeof(RUNTIME_AREA) < 0x1000);
#endif

#define STACK_ALIGN(type, name, cnt, alignment)		UCHAR _al__##name[((sizeof(type)*(cnt)) + (alignment) + (((sizeof(type)*(cnt))%(alignment)) > 0 ? ((alignment) - ((sizeof(type)*(cnt))%(alignment))) : 0))]; \
													type *name = (type*)(((ULONG)(_al__##name)) + ((alignment) - (((ULONG)(_al__##name))&((alignment)-1))))
#define IDENTIFIER_MIO "VME"