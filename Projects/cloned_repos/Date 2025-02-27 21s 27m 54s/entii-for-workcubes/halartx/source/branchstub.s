// Branch to reset stub.

#include <kxppc.h>

	LEAF_ENTRY(HalpBranchToResetStub)
	
	// Move passed in address to srr0 ready to jump there
	mtsrr0 r.3
	mfmsr r.3
	// disable MSR_ILE
	rlwinm r.3, r.3, 0, 16, 14
	mtmsr r.3
	// disable MSR_LE into srr1
	rlwinm r.3, r.3, 0, 0, 30
	mtsrr1 r.3
	// jump to reset stub.
	rfi
	
	LEAF_EXIT(HalpBranchToResetStub)