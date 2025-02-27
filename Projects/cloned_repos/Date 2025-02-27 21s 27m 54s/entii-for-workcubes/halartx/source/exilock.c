// EXI lock code.
#define DEVL 1
#include "halp.h"
#include "exiapi.h"

enum {
	COUNT_EMERGENCY_BLOCKS = 32
};

// Wait control block for the EXI channel locks.
typedef struct _EXI_WAIT_CONTROL_BLOCK {
	KDEVICE_QUEUE_ENTRY DeviceQueueEntry;
	HAL_EXI_LOCK_CALLBACK Callback;
	PVOID Context;
} EXI_WAIT_CONTROL_BLOCK, *PEXI_WAIT_CONTROL_BLOCK;

typedef struct _EXI_LOCK_WORK_ITEM {
	WORK_QUEUE_ITEM WorkItem;
	ULONG Channel;
	HAL_EXI_LOCK_CALLBACK Callback;
	PVOID Context;
} EXI_LOCK_WORK_ITEM, *PEXI_LOCK_WORK_ITEM;

typedef struct _EXI_UNLOCK_WORK_ITEM {
	WORK_QUEUE_ITEM WorkItem;
	ULONG Channel;
} EXI_UNLOCK_WORK_ITEM, *PEXI_UNLOCK_WORK_ITEM;

static KDEVICE_QUEUE s_ExiLockQueues[EXI_CHANNEL_COUNT] = {0};
static EXI_WAIT_CONTROL_BLOCK s_EmergencyBlocks[COUNT_EMERGENCY_BLOCKS] = {0};

BOOLEAN HalExiLocked(ULONG channel) {
	if (channel >= EXI_CHANNEL_COUNT) return FALSE;
	// This offset has stayed constant from NT 3.1 to NT 6.0 :)
	return s_ExiLockQueues[channel].Busy;
}

static void ExipUnlock(ULONG channel) {
	EXI_LOCK_ACTION Action = ExiUnlock;
	while (Action == ExiUnlock) {
		KIRQL CurrentIrql;
		KeRaiseIrql(DISPATCH_LEVEL, &CurrentIrql);
		PEXI_WAIT_CONTROL_BLOCK Block = (PEXI_WAIT_CONTROL_BLOCK) KeRemoveDeviceQueue(&s_ExiLockQueues[channel]);
		KeLowerIrql(CurrentIrql);
		
		if (Block == NULL) break;
		
		Action = Block->Callback(channel, Block->Context);
		
		if (Block >= &s_EmergencyBlocks[0] && Block < &s_EmergencyBlocks[COUNT_EMERGENCY_BLOCKS]) {
			Block->Callback = NULL;
		} else ExFreePool(Block);
	}
}

static void ExipUnlockWorkRoutine(PEXI_UNLOCK_WORK_ITEM Parameter) {
	// Grab the parameters.
	ULONG Channel = Parameter->Channel;
	// Free the work item
	ExFreePool(Parameter);
	
	ExipUnlock(Channel);
}

static NTSTATUS ExipLock(ULONG channel, HAL_EXI_LOCK_CALLBACK callback, PVOID context) {
	// Allocate a control block to store the callback.
	PEXI_WAIT_CONTROL_BLOCK Block = (PEXI_WAIT_CONTROL_BLOCK) ExAllocatePool(NonPagedPool, sizeof(EXI_WAIT_CONTROL_BLOCK));
	if (Block == NULL) {
		// Pick out the first emergency block with a null callback.
		for (ULONG i = 0; i < COUNT_EMERGENCY_BLOCKS; i++) {
			if (s_EmergencyBlocks[i].Callback != NULL) continue;
			Block = &s_EmergencyBlocks[i];
			break;
		}
		// If none were found, return insufficient resources.
		if (Block == NULL) return STATUS_INSUFFICIENT_RESOURCES;
	}
	
	RtlZeroMemory(Block, sizeof(*Block));
	Block->Callback = callback;
	Block->Context = context;
	
	// Go to DISPATCH_LEVEL when calling device queue related functions
	KIRQL CurrentIrql;
	KeRaiseIrql(DISPATCH_LEVEL, &CurrentIrql);
	BOOLEAN Result = KeInsertDeviceQueue(&s_ExiLockQueues[channel], &Block->DeviceQueueEntry);
	KeLowerIrql(CurrentIrql);
	
	if (!Result) {
		// The controller is not busy.
		// While a lock callback is present, call it and free the block.
		do {
			EXI_LOCK_ACTION Action = Block->Callback(channel, Block->Context);
			
			if (Block >= &s_EmergencyBlocks[0] && Block < &s_EmergencyBlocks[COUNT_EMERGENCY_BLOCKS]) {
				Block->Callback = NULL;
			} else ExFreePool(Block);
			
			if (Action == ExiKeepLocked) break;
			
			KeRaiseIrql(DISPATCH_LEVEL, &CurrentIrql);
			Block = (PEXI_WAIT_CONTROL_BLOCK) KeRemoveDeviceQueue(&s_ExiLockQueues[channel]);
			KeLowerIrql(CurrentIrql);
		} while (Block != NULL);
		
		return STATUS_SUCCESS;
	}
	
	DbgPrint("EXILOCK: caller at %08x wanted locked chan%d\n", __builtin_return_address(0), channel);
	return STATUS_PENDING;
}

static void ExipLockWorkRoutine(PEXI_LOCK_WORK_ITEM Parameter) {
	// Grab the parameters.
	ULONG Channel = Parameter->Channel;
	HAL_EXI_LOCK_CALLBACK Callback = Parameter->Callback;
	PVOID Context = Parameter->Context;
	// Free the work item
	ExFreePool(Parameter);
	
	ExipLock(Channel, Callback, Context);
}

NTSTATUS HalExiUnlock(ULONG channel) {
	if (channel >= EXI_CHANNEL_COUNT) return STATUS_INVALID_PARAMETER;
	
	KIRQL CurrentIrql = KeGetCurrentIrql();
	
	if (CurrentIrql > DISPATCH_LEVEL) return STATUS_LOCK_NOT_GRANTED;
	
	if (CurrentIrql == DISPATCH_LEVEL) {
		// Currently at DISPATCH_LEVEL, go through the work pool
		// Allocate the work item
		PEXI_UNLOCK_WORK_ITEM WorkItem = (PEXI_UNLOCK_WORK_ITEM)
			ExAllocatePool(NonPagedPool, sizeof(EXI_UNLOCK_WORK_ITEM));
		
		if (WorkItem == NULL) return STATUS_INSUFFICIENT_RESOURCES;
		
		// Fill in the parameters
		WorkItem->Channel = channel;
		
		// Initialise the ExWorkItem
		ExInitializeWorkItem(
			&WorkItem->WorkItem,
			(PWORKER_THREAD_ROUTINE) ExipUnlockWorkRoutine,
			WorkItem
		);
		
		// Queue it
		ExQueueWorkItem(&WorkItem->WorkItem, DelayedWorkQueue);
		return STATUS_PENDING;
	}
	
	ExipUnlock(channel);
	return STATUS_SUCCESS;
}

NTSTATUS HalExiUnlockNonpaged(ULONG channel) {
	if (channel >= EXI_CHANNEL_COUNT) return STATUS_INVALID_PARAMETER;
	
	KIRQL CurrentIrql = KeGetCurrentIrql();
	
	if (CurrentIrql > DISPATCH_LEVEL) return STATUS_LOCK_NOT_GRANTED;
	
	ExipUnlock(channel);
	return STATUS_SUCCESS;
}

NTSTATUS HalExiLock(ULONG channel, HAL_EXI_LOCK_CALLBACK callback, PVOID context) {
	if (channel >= EXI_CHANNEL_COUNT) return STATUS_INVALID_PARAMETER;
	
	KIRQL CurrentIrql = KeGetCurrentIrql();
	
	if (CurrentIrql > DISPATCH_LEVEL) return STATUS_LOCK_NOT_GRANTED;
	
	if (CurrentIrql == DISPATCH_LEVEL) {
		// Currently at DISPATCH_LEVEL, go through the work pool
		// Allocate the work item
		PEXI_LOCK_WORK_ITEM WorkItem = (PEXI_LOCK_WORK_ITEM)
			ExAllocatePool(NonPagedPool, sizeof(EXI_LOCK_WORK_ITEM));
		
		if (WorkItem == NULL) return STATUS_INSUFFICIENT_RESOURCES;
		
		// Fill in the parameters
		WorkItem->Channel = channel;
		WorkItem->Callback = callback;
		WorkItem->Context = context;
		
		// Initialise the ExWorkItem
		ExInitializeWorkItem(
			&WorkItem->WorkItem,
			(PWORKER_THREAD_ROUTINE) ExipLockWorkRoutine,
			WorkItem
		);
		
		// Queue it
		ExQueueWorkItem(&WorkItem->WorkItem, DelayedWorkQueue);
		return STATUS_PENDING;
	}
	
	return ExipLock(channel, callback, context);
}

NTSTATUS HalExiLockNonpaged(ULONG channel, HAL_EXI_LOCK_CALLBACK callback, PVOID context) {
	if (channel >= EXI_CHANNEL_COUNT) return STATUS_INVALID_PARAMETER;
	
	KIRQL CurrentIrql = KeGetCurrentIrql();
	
	if (CurrentIrql > DISPATCH_LEVEL) return STATUS_LOCK_NOT_GRANTED;
	
	return ExipLock(channel, callback, context);
}

void HalExiLockInit(void) {
	for (ULONG i = 0; i < EXI_CHANNEL_COUNT; i++) {
		KeInitializeDeviceQueue(&s_ExiLockQueues[i]);
	}
}