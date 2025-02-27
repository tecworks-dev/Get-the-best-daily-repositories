#pragma once

enum { // FPF = "Flipper Flag"
	FPF_IN_EMULATOR = ARC_BIT(0), // System is Dolphin emulator
	FPF_IS_VEGAS = ARC_BIT(1), // System is Vegas (RVL)
	FPF_IS_LATTE = ARC_BIT(2), // System is Latte (Cafe)
};

typedef struct _HW_DESCRIPTION {
	ULONG MemoryLength[2]; // Length of physical memory. (Splash/Napa, DDR)
	ULONG DdrIpcBase, DdrIpcLength; // Start and length of IOS IPC buffer in DDR
	ULONG DecrementerFrequency; // Decrementer frequency.
	ULONG RtcBias; // RTC counter bias.
	ULONG FpFlags; // FPF_* bit flags.

	// Framebuffer details.
	ULONG FrameBufferBase; // Base address of frame buffer.
	ULONG FrameBufferLength; // Length of frame buffer in video RAM.
	ULONG FrameBufferWidth; // Display width
	ULONG FrameBufferHeight; // Display height
	ULONG FrameBufferStride; // Number of bytes per line.
	
	// GX FIFO memory.
	ULONG GxFifoBase; // Base address of GX FIFO (length is 64KB)
	
	ULONG Padding;
} HW_DESCRIPTION, *PHW_DESCRIPTION;

_Static_assert((sizeof(HW_DESCRIPTION) % sizeof(unsigned long long)) == 0);
