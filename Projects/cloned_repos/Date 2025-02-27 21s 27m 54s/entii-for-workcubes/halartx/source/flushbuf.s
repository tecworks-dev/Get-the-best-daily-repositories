// Implement KeFlushWriteBuffer export

#include "kxppc.h"

// KeFlushWriteBuffer: eieio + sync (full memory barrier)

	LEAF_ENTRY(KeFlushWriteBuffer)

	eieio
	sync
	
	LEAF_EXIT(KeFlushWriteBuffer)