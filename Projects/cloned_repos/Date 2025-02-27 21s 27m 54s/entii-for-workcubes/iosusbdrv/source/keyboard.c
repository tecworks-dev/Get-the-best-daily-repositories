// Keyboard driver NT side.
// Includes conversion tables/etc from USB HID scancodes to PS/2 scancodes.
// This was already done for NT5 by kbdhid.sys, hidparse.sys
// so just reimplement that.
// That said, we only support boot protocol.
#define DEVL 1
#include <ntddk.h>
#include <kbdmou.h>
#include <stdio.h>
#include "keyboard.h"
#include "zwdd.h"
#define SIZEOF_ARRAY(x) (sizeof((x)) / sizeof((x)[0]))
#define RtlCopyMemory(Destination,Source,Length) memcpy((Destination),(Source),(Length))

static BOOLEAN s_CreatedOneKeyboardDevice = FALSE;

enum {
	CONVERT_TYPE_KEY_UP,
	CONVERT_TYPE_KEY_DOWN
};

enum {
	SECONDARY_PAD,
	SECONDARY_MOD,
	SECONDARY_VEN,
	SECONDARY_PRSC
};

enum {
	KEY_MODIFIER_CAPSLOCK = BIT(8),
	KEY_MODIFIER_SCROLLLOCK = BIT(9),
	KEY_MODIFIER_NUMLOCK = BIT(10),

	KEYLED_NUM = BIT(0),
	KEYLED_CAPS = BIT(1),
	KEYLED_SCROLL = BIT(2)
};

typedef enum _KEYBOARD_SCAN_STATE {
    Normal,
    GotE0,
    GotE1
} KEYBOARD_SCAN_STATE, *PKEYBOARD_SCAN_STATE;

// Define the conversion tables.
// This comes from kbdhid.sys:

static const INDICATOR_LIST sc_IndicatorList[] = {
        {0x3A, KEYBOARD_CAPS_LOCK_ON},
        {0x45, KEYBOARD_NUM_LOCK_ON},
        {0x46, KEYBOARD_SCROLL_LOCK_ON}
};

// These come from hidparse.sys:

#define INVALID 0xFF
#define TABLE_SUB(x,b) (((x) << 8) | (b))
#define PAD(x) TABLE_SUB(x, 0xF0)
#define MOD(x) TABLE_SUB(x, 0xF1)
#define VEN(x) TABLE_SUB(x, 0xF2)
#define PRSC(x) TABLE_SUB(x, 0xF3)

static const ULONG sc_UsbToPs2Table[] = {

INVALID,  INVALID,  INVALID,  INVALID,  0x1E,     0x30,     0x2E,    0x20, 
0x12,     0x21,     0x22,     0x23,     0x17,     0x24,     0x25,    0x26, 
0x32,     0x31,     0x18,     0x19,     0x10,     0x13,     0x1F,    0x14, 
0x16,     0x2F,     0x11,     0x2D,     0x15,     0x2C,     0x02,    0x03, 
0x04,     0x05,     0x06,     0x07,     0x08,     0x09,     0x0A,    0x0B, 
0x1C,     0x01,     0x0E,     0x0F,     0x39,     0x0C,     0x0D,    0x1A, 
0x1B,     0x2B,     0x2B,     0x27,     0x28,     0x29,     0x33,    0x34, 
0x35,     MOD(8),   0x3B,     0x3C,     0x3D,     0x3E,     0x3F,    0x40, 
0x41,     0x42,     0x43,     0x44,     0x57,     0x58,     PRSC(0), MOD(9), 
0x451DE1, PAD(0),   PAD(1),   PAD(2),   PAD(3),   PAD(4),   PAD(5),  PAD(6), 
PAD(7),   PAD(8),   PAD(9),   MOD(0xA), 0x35E0,   0x37,     0x4A,    0x4E, 
0x1CE0,   0x4F,     0x50,     0x51,     0x4B,     0x4C,     0x4D,    0x47, 
0x48,     0x49,     0x52,     0x53,     0x56,     0x5DE0,   0x5EE0,  0x59, 
0x64,     0x65,     0x66,     0x67,     0x68,     0x69,     0x6A,    0x6B, 
0x6C,     0x6D,     0x6E,     0x76,     INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  0x7E,     INVALID, 0x73, 
0x70,     0x7D,     0x79,     0x7B,     0x5C,     INVALID,  INVALID, INVALID,
VEN(0),   VEN(1),   0x78,     0x77,     0x76,     INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
MOD(0),   MOD(1),   MOD(2),   MOD(3),   MOD(4),   MOD(5),   MOD(6),  MOD(7), 
INVALID,  0x5EE0,   0x5FE0,   0x63E0,   INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,
INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID,  INVALID, INVALID,

};

// 0x49 to 0x52
static const ULONG sc_UsbToPs2PadTable[] = {

0x52E0, 0x47E0, 0x49E0, 0x53E0, 0x4FE0, 0x51E0, 0x4DE0, 0x4BE0,
0x50E0, 0x48E0

};

// Bit order for modifiers, followed by capslock, scrolllock, numlock
static const ULONG sc_UsbToPs2ModTable[] = {

// LCTRL, LSHIFT, LALT, LWIN,   RCTRL,  RSHIFT, RALT,   RWIN
   0x1D,  0x2A,   0x38, 0x5BE0, 0x1DE0, 0x36,   0x38E0, 0x5CE0,

0x3A, 0x46, 0x45

};

// 0x90,0x91
static const ULONG sc_UsbToPs2VenTable[] = {

0xF2, 0xF1

};

PDEVICE_OBJECT KbdDeviceObject = NULL;
static CONNECT_DATA s_ClassConnection = {0};

// USB HID driver puts this in device extension.
// We will only ever handle ONE keyboard device so let's not bother.
// (that is, the first known USB keyboard low-level, or whatever IOS gives us)
static USB_KBD_REPORT s_OldReport = {0};
static ULONG s_ModifierState = 0;
static KEYBOARD_INPUT_DATA s_InputData = {0};
static KEYBOARD_SCAN_STATE s_ScanState = {0};
static KEYBOARD_INDICATOR_PARAMETERS s_Indicators = {0};
static ULONG s_EnableCount = 0;


static void KbdpSendToClass(const UCHAR * Ps2ScanCodes, ULONG Length) {
	PKEYBOARD_INPUT_DATA InputDataEnd = &s_InputData;
	InputDataEnd++;
	for (ULONG Index = 0; Index < Length; Index++) {
		UCHAR ScanCode = Ps2ScanCodes[Index];
		
		if (ScanCode == 0xFF) {
			s_InputData.MakeCode = KEYBOARD_OVERRUN_MAKE_CODE;
			s_InputData.Flags = 0;
			s_ScanState = Normal;
			continue;
		}
		if (s_ScanState == Normal) {
			// Change state if needed.
			if (ScanCode == 0xE0) {
				s_InputData.Flags |= KEY_E0;
				s_ScanState = GotE0;
				continue;
			} else if (ScanCode == 0xE1) {
				s_InputData.Flags |= KEY_E1;
				s_ScanState = GotE1;
				continue;
			}
		} else if (s_ScanState != GotE0 && s_ScanState != GotE1) {
			// should not happen?!
			return;
		}
		
		// Strip the high bit when setting the input scancode.
		s_InputData.MakeCode = ScanCode & ~0x80;
		// If the high bit was set, it was a break code.
		if ((ScanCode & 0x80) != 0) {
			s_InputData.Flags |= KEY_BREAK;
		}
		s_ScanState = Normal;
		
		// We have a full scancode.
		// If enabled, call the input callback for this single element.
		// Callback needs to be called at DISPATCH_LEVEL
		if (s_EnableCount && s_ClassConnection.ClassService != NULL) {
			PSERVICE_CALLBACK_ROUTINE OnKbdInput = (PSERVICE_CALLBACK_ROUTINE)
				s_ClassConnection.ClassService;
			KIRQL OldIrql;
			KeRaiseIrql(DISPATCH_LEVEL, &OldIrql);
			ULONG ElementsRead;
			OnKbdInput(
				s_ClassConnection.ClassDeviceObject,
				&s_InputData,
				InputDataEnd,
				&ElementsRead
			);
			KeLowerIrql(OldIrql);
			s_InputData.Flags = 0;
			if (ElementsRead != 1) {
				// should never happen
			}
		} else {
			s_InputData.Flags = 0;
		}
	}
}

static void KbdpMixTypeAndSend(ULONG Code, ULONG Type) {
	// Treat 32-bit input as bytes.
	PUCHAR pCode = (PUCHAR)&Code;
	// If there are no bytes to send, then nothing needs to be done.
	if (pCode[0] == 0) return;
	ULONG Length = 0;
	for (ULONG i = 0; i < sizeof(Code); i++) {
		if (pCode[i] == 0) break;
		Length++;
		// Bit 7 means key up (extended scancodes already have bit7 set)
		// So if this is key-up, then set that bit on each used byte.
		if (Type == CONVERT_TYPE_KEY_UP) pCode[i] |= 0x80;
	}
	// Send the bytes.
	KbdpSendToClass(pCode, Length);
}

static void KbdpUsbToPs2Modifier(UCHAR Index, ULONG Type) {
	if (Type == CONVERT_TYPE_KEY_UP) {
		s_ModifierState &= ~(1 << Index);
	} else { // KEY_DOWN
		s_ModifierState |= (1 << Index);
	}
	
	KbdpMixTypeAndSend(sc_UsbToPs2ModTable[Index], Type);
}

static void KbdpUsbToPs2Modifiers(UCHAR Modifiers, ULONG Type) {
	// Loop for each bit in Modifiers.
	// If no more bits are set, nothing more needs to be done.
	for (ULONG Bit = 0; Bit < 8 && Modifiers != 0; Bit++, Modifiers >>= 1) {
		if ((Modifiers & 1) == 0) continue;
		KbdpUsbToPs2Modifier(Bit, Type);
	}
}

static void KbdpUsbToPs2(PUCHAR Buffer, ULONG Length, UCHAR ModifiersChanged, ULONG Type) {
	// first, deal with modifiers
	KbdpUsbToPs2Modifiers(ModifiersChanged, Type);
	for (ULONG i = 0; i < Length; i++) {
		UCHAR Usb = Buffer[i];
		if (Usb == 0) break;
		// First, check the initial lookup table.
		ULONG Ps2 = sc_UsbToPs2Table[Usb];
		// If lookup table gives a low byte of 0xFx,
		// a secondary table needs to be used.
		if ((Ps2 & 0xF0) != 0xF0) {
			// No secondary table needed. Just send the bytes.
			KbdpMixTypeAndSend(Ps2, Type);
			continue;
		}
		// Which secondary table?
		UCHAR SecondaryTable = Ps2 & 0xF;
		UCHAR SecondaryIndex = (UCHAR)(Ps2 >> 8);
		static const UCHAR sc_PrintScreenHalf[] = { 0xE0, 0x2A };
		static const UCHAR sc_PrintScreenHalfUp[] = { 0xE0, 0xAA };
		if (SecondaryTable == SECONDARY_PAD) {
			// Numpad.
			// Go into other mode.
			if ((s_ModifierState & KEY_MODIFIER_NUMLOCK) != 0 && Type == CONVERT_TYPE_KEY_DOWN) {
				KbdpSendToClass(sc_PrintScreenHalf, sizeof(sc_PrintScreenHalf));
			}
			KbdpMixTypeAndSend(sc_UsbToPs2PadTable[SecondaryIndex], Type);
			if ((s_ModifierState & KEY_MODIFIER_NUMLOCK) != 0 && Type == CONVERT_TYPE_KEY_UP) {
				KbdpSendToClass(sc_PrintScreenHalfUp, sizeof(sc_PrintScreenHalfUp));
			}
			continue;
		}
		if (SecondaryTable == SECONDARY_MOD) {
			// Additional modifier (capslock/scrolllock/numlock)
			UCHAR BitFirst = SecondaryIndex + 16;
			if (Type == CONVERT_TYPE_KEY_UP) {
				// Clear the first-press bit, leaving the sticky bit as was.
				s_ModifierState &= ~(1 << BitFirst);
			} else if ((s_ModifierState & (1 << BitFirst)) == 0) { // also KEY_DOWN
				// First-press bit was not set. Set it now.
				s_ModifierState |= (1 << BitFirst);
				// Toggle the sticky-bit.
				s_ModifierState ^= (1 << SecondaryIndex);
			}
			// Send the keypress.
			KbdpMixTypeAndSend(sc_UsbToPs2ModTable[SecondaryIndex], Type);
			continue;
		}
		if (SecondaryTable == SECONDARY_VEN) {
			// Don't send key-up for this secondary-type.
			if (Type == CONVERT_TYPE_KEY_UP) continue;
			KbdpMixTypeAndSend(sc_UsbToPs2VenTable[SecondaryIndex], Type);
			continue;
		}
		if (SecondaryTable == SECONDARY_PRSC) {
			// Print screen needs special handling.
			if ((s_ModifierState & KEY_MODIFIER_ALT) != 0) {
				// Alt + print screen
				KbdpMixTypeAndSend(0x54, Type);
				continue;
			}
			if ((s_ModifierState & (KEY_MODIFIER_CTRL | KEY_MODIFIER_SHIFT)) != 0) {
				// Ctrl or shift + print screen
				KbdpMixTypeAndSend(0x37E0, Type);
				continue;
			}
			// No modifier keys down, go into other mode first.
			if (Type == CONVERT_TYPE_KEY_DOWN) {
				KbdpSendToClass(sc_PrintScreenHalf, sizeof(sc_PrintScreenHalf));
			}
			KbdpMixTypeAndSend(0x37E0, Type);
			if (Type == CONVERT_TYPE_KEY_UP) {
				KbdpSendToClass(sc_PrintScreenHalfUp, sizeof(sc_PrintScreenHalfUp));
			}
		}
	}
}


void KbdReadComplete(PUSB_KBD_REPORT Report) {
	if (!s_EnableCount) return;
	if (s_ClassConnection.ClassService == NULL) return;
	// We have a single input report.
	if (Report->KeyCode[0] == KEY_ERROR_OVF) {
		// Overflow condition, just discard this report.
		return;
	}
	
	BOOLEAN ReportIsDifferent = FALSE;
	// Compare old report to new report.
	// Modifiers first.
	UCHAR ModifiersDown = 0, ModifiersUp = 0;
	if (Report->Modifiers != s_OldReport.Modifiers) {
		ReportIsDifferent = TRUE;
		// Mask out the modifiers that were already down.
		ModifiersDown = Report->Modifiers & ~s_OldReport.Modifiers;
		// Mask out the modifiers that were down and are no longer.
		ModifiersUp = ~Report->Modifiers & s_OldReport.Modifiers;
	}
	UCHAR KeysDown[sizeof(Report->KeyCode)];
	UCHAR KeysUp[sizeof(Report->KeyCode)];
	UCHAR IndexDown = 0, IndexUp = 0;
	// Get the keys that were released.
	for (UCHAR i = 0; i < sizeof(Report->KeyCode); i++) {
		if (s_OldReport.KeyCode[i] == 0) {
			// No key pressed.
			break;
		}
		if (s_OldReport.KeyCode[i] < 0x04) {
			// Error.
			continue;
		}
		// If this key was NOT in the new list, it was released
		BOOLEAN InNewList = FALSE;
		for (UCHAR newI = 0; newI < sizeof(Report->KeyCode); newI++) {
			if (Report->KeyCode[newI] == 0) break;
			if (Report->KeyCode[newI] == s_OldReport.KeyCode[i]) {
				InNewList = TRUE;
				break;
			}
		}
		if (InNewList) continue;
		ReportIsDifferent = TRUE;
		KeysUp[IndexUp] = s_OldReport.KeyCode[i];
		IndexUp++;
	}
	// Get the keys that were pressed.
	for (UCHAR i = 0; i < sizeof(Report->KeyCode); i++) {
		if (Report->KeyCode[i] == 0) {
			// No key pressed.
			break;
		}
		if (Report->KeyCode[i] < 0x04) {
			// Error.
			continue;
		}
		// If this key was in the old list, it was already pressed.
		BOOLEAN InOldList = FALSE;
		for (UCHAR oldI = 0; oldI < sizeof(Report->KeyCode); oldI++) {
			if (s_OldReport.KeyCode[oldI] == 0) break;
			if (s_OldReport.KeyCode[oldI] == Report->KeyCode[i]) {
				InOldList = TRUE;
				break;
			}
		}
		if (InOldList) continue;
		ReportIsDifferent = TRUE;
		KeysDown[IndexDown] = Report->KeyCode[i];
		IndexDown++;
	}
	
	// The difference between the current and previous reports is now known.
	// If there was no difference, do nothing.
	if (!ReportIsDifferent) return;
	
	// Send keys-up.
	KbdpUsbToPs2(KeysUp, IndexUp, ModifiersUp, CONVERT_TYPE_KEY_UP);
	// Send keys-down.
	KbdpUsbToPs2(KeysDown, IndexDown, ModifiersDown, CONVERT_TYPE_KEY_DOWN);
	// Set old report to current report.
	RtlCopyMemory(&s_OldReport, Report, sizeof(*Report));
	// TODO: timer for repeating keys?
}

static UCHAR KbdpGpioOutPs2ToUsb(void) {
	UCHAR Usb = 0;
	USHORT Ps2 = s_Indicators.LedFlags;
	if ((Ps2 & KEYBOARD_SCROLL_LOCK_ON) != 0) {
		Usb |= KEYLED_SCROLL;
	}
	if ((Ps2 & KEYBOARD_NUM_LOCK_ON) != 0) {
		Usb |= KEYLED_NUM;
	}
	if ((Ps2 & KEYBOARD_CAPS_LOCK_ON) != 0) {
		Usb |= KEYLED_CAPS;
	}
	return Usb;
}

static NTSTATUS KbdpIoctl(
	PDEVICE_OBJECT Device,
	PIRP Irp
) {
	// assumption: Device == KbdDeviceObject
	PIO_STACK_LOCATION Stack = IoGetCurrentIrpStackLocation(Irp);
	__auto_type Params = &Stack->Parameters.DeviceIoControl;
	
	switch (Params->IoControlCode) {
	case IOCTL_INTERNAL_KEYBOARD_CONNECT:
		// Keyboard class driver is giving us its device object and callback.
		if (s_ClassConnection.ClassService != NULL) {
			// ...but it already did!
			return STATUS_SHARING_VIOLATION;
		}
		if (Params->InputBufferLength < sizeof(s_ClassConnection)) {
			// ...but the buffer was too small
			return STATUS_INVALID_PARAMETER;
		}
		RtlCopyMemory(&s_ClassConnection, Params->Type3InputBuffer, sizeof(s_ClassConnection));
		return STATUS_SUCCESS;
	case IOCTL_INTERNAL_KEYBOARD_DISCONNECT:
		// Keyboard class driver is telling us to stop using its callback.
		// But nobody ever implements this, so why should we bother?
		return STATUS_NOT_IMPLEMENTED;
	case IOCTL_INTERNAL_KEYBOARD_ENABLE:
		// Keyboard class driver is telling us to increment the enable count.
		if (s_EnableCount == 0xFFFFFFFF) return STATUS_DEVICE_DATA_ERROR;
		InterlockedIncrement(&s_EnableCount);
		return STATUS_SUCCESS;
	case IOCTL_INTERNAL_KEYBOARD_DISABLE:
		// Keyboard class driver is telling us to decrement the enable count.
		if (s_EnableCount == 0) return STATUS_DEVICE_DATA_ERROR;
		InterlockedDecrement(&s_EnableCount);
		return STATUS_SUCCESS;
	case IOCTL_KEYBOARD_QUERY_ATTRIBUTES:
		// Caller wants keyboard attributes which are basically hardcoded.
		if (Params->OutputBufferLength < sizeof(KEYBOARD_ATTRIBUTES)) {
			return STATUS_BUFFER_TOO_SMALL;
		}
		{
			PKEYBOARD_ATTRIBUTES Attributes = (PKEYBOARD_ATTRIBUTES)
				Irp->AssociatedIrp.SystemBuffer;
			RtlZeroMemory(Attributes, sizeof(*Attributes));
			Attributes->KeyboardIdentifier.Type = 81; // hardcoded in kbdhid.sys
			Attributes->KeyboardMode = 1; // set 1?
			Attributes->NumberOfFunctionKeys = 12; // claim to have F1-F12
			Attributes->NumberOfIndicators = 3; // GPIO outs for num lock, caps lock, scroll lock
			Attributes->NumberOfKeysTotal = 101; // 101 keys
			Attributes->InputDataQueueLength = 1;
		}
		return STATUS_SUCCESS;
	case IOCTL_KEYBOARD_QUERY_INDICATOR_TRANSLATION:
		// Caller wants list of PS2 scancodes->indicator bitflags.
		{
			ULONG LengthArr = sizeof(INDICATOR_LIST) * SIZEOF_ARRAY(sc_IndicatorList);
			ULONG LengthBase = __builtin_offsetof(KEYBOARD_INDICATOR_TRANSLATION, IndicatorList);
			ULONG Length = LengthBase + LengthArr;
			if (Params->OutputBufferLength < Length) {
				return STATUS_BUFFER_TOO_SMALL;
			}
			PKEYBOARD_INDICATOR_TRANSLATION Translation =
				(PKEYBOARD_INDICATOR_TRANSLATION)
				Irp->AssociatedIrp.SystemBuffer;
			Translation->NumberOfIndicatorKeys = SIZEOF_ARRAY(sc_IndicatorList);
			RtlCopyMemory(Translation->IndicatorList, sc_IndicatorList, LengthArr);
			return STATUS_SUCCESS;
		}
	case IOCTL_KEYBOARD_QUERY_INDICATORS:
		// Caller wants current keyboard out-GPIO state.
		if (Params->OutputBufferLength < sizeof(s_Indicators)) {
			return STATUS_BUFFER_TOO_SMALL;
		}
		RtlCopyMemory(Irp->AssociatedIrp.SystemBuffer, &s_Indicators, sizeof(s_Indicators));
		return STATUS_SUCCESS;
	case IOCTL_KEYBOARD_SET_INDICATORS:
		// Caller wants to set new keyboard out-GPIO state.
		if (Params->InputBufferLength < sizeof(s_Indicators)) {
			return STATUS_INVALID_PARAMETER;
		}
		RtlCopyMemory(&s_Indicators, Irp->AssociatedIrp.SystemBuffer, sizeof(s_Indicators));
		{
			UCHAR UsbIndicators = KbdpGpioOutPs2ToUsb();
			NTSTATUS Status = IukSetLeds(UsbIndicators);
			if (NT_SUCCESS(Status) || Status != STATUS_NO_SUCH_DEVICE) return Status;
			return UlkSetLeds(UsbIndicators);
		}
	case IOCTL_KEYBOARD_SET_TYPEMATIC:
		// Caller wants to update keyboard timer-rate.
		// Let's just not implement this for now?
		return STATUS_NOT_IMPLEMENTED;
	default:
		return STATUS_INVALID_DEVICE_REQUEST;
	}
}

NTSTATUS KbdIoctl(
	PDEVICE_OBJECT Device,
	PIRP Irp
) {
	NTSTATUS Status = KbdpIoctl(Device, Irp);
	Irp->IoStatus.Status = Status;
	IoCompleteRequest(Irp, IO_NO_INCREMENT);
	return Status;
}

#if 0
typedef struct _REGVALUE_WORK_ITEM {
	WORK_QUEUE_ITEM WorkItem;
	KEVENT Event;
	PWSTR Path;
	PVOID Value;
	ULONG Length;
	NTSTATUS Status;
} REGVALUE_WORK_ITEM, *PREGVALUE_WORK_ITEM;

static void RtlWriteRegistryValueInOtherThread(PREGVALUE_WORK_ITEM WorkItem) {
	static USHORT s_KeyboardPortKey[] = {
				'K', 'e', 'y', 'b', 'o', 'a', 'r', 'd',
				'P', 'o', 'r', 't', 0
			};
			
	HalDisplayString("RtlWriteRegistryValue other\n");
	WorkItem->Status = RtlWriteRegistryValue(
				RTL_REGISTRY_DEVICEMAP,
				s_KeyboardPortKey,
				WorkItem->Path,
				REG_SZ,
				WorkItem->Value,
				WorkItem->Length
	);
	
	KeSetEvent(&WorkItem->Event, 0, FALSE);
}

static NTSTATUS KbdWriteRegistryValue(PWSTR Path, PVOID Value, ULONG Length) {
	// Allocate the work item
	PREGVALUE_WORK_ITEM WorkItem = (PREGVALUE_WORK_ITEM)
		ExAllocatePool(NonPagedPool, sizeof(REGVALUE_WORK_ITEM));
		
	if (WorkItem == NULL) return STATUS_NO_MEMORY;
	WorkItem->Path = Path;
	WorkItem->Value = Value;
	WorkItem->Length = Length;
	
	KeInitializeEvent(&WorkItem->Event, NotificationEvent, FALSE);
	
	// Initialise the ExWorkItem
	ExInitializeWorkItem(
		&WorkItem->WorkItem,
		(PWORKER_THREAD_ROUTINE) RtlWriteRegistryValueInOtherThread,
		WorkItem
	);
	
	// Queue it
	ExQueueWorkItem(&WorkItem->WorkItem, CriticalWorkQueue);
	
	// Wait on it
	KeWaitForSingleObject(&WorkItem->Event, Executive, KernelMode, FALSE, NULL);
	
	// Grab the status
	NTSTATUS Status = WorkItem->Status;
	
	// Free the work item
	ExFreePool(WorkItem);
	
	return Status;
}
#endif

NTSTATUS KbdInit(PDRIVER_OBJECT Driver, PUNICODE_STRING RegistryPath) {
	// Create only one keyboard device.
	if (KbdDeviceObject != NULL) return STATUS_SHARING_VIOLATION;
	// We expect no other keyboard devices to have been created;
	// but just in case, loop through all of them anyways.
	NTSTATUS Status = STATUS_UNSUCCESSFUL;
	for (ULONG i = 0; i < KEYBOARD_PORTS_MAXIMUM; i++) {
		// Generate the NT object path name for this device.
		UCHAR NtName[256];
		//HalDisplayString("snprintf\n");
		_snprintf(NtName, sizeof(NtName), "\\Device\\KeyboardPort%d", i);
		// Wrap an ANSI string around it.
		STRING NtNameStr;
		//HalDisplayString("RtlInitString\n");
		RtlInitString(&NtNameStr, NtName);
		// Convert to unicode.
		UNICODE_STRING NtNameUs;
		//HalDisplayString("RtlAnsiStringToUnicodeString\n");
		Status = RtlAnsiStringToUnicodeString(&NtNameUs, &NtNameStr, TRUE);
		if (!NT_SUCCESS(Status)) continue;
		// Create the device object.
		//HalDisplayString("IoCreateDevice\n");
		Status = IoCreateDevice(Driver, 0, &NtNameUs, FILE_DEVICE_8042_PORT, 0, FALSE, &KbdDeviceObject);
		if (!NT_SUCCESS(Status)) {
			//HalDisplayString("RtlFreeUnicodeString\n");
			RtlFreeUnicodeString(&NtNameUs);
			continue;
		}
		// Created it.
		// If a USB keyboard is plugged in prioritise that over a SI device.
		//if (i != 0) s_CreatedOneKeyboardDevice = TRUE;
		if (!s_CreatedOneKeyboardDevice) {
			static USHORT s_KeyboardPortKey[] = {
				'K', 'e', 'y', 'b', 'o', 'a', 'r', 'd',
				'P', 'o', 'r', 't', 0
			};
			
			//CHAR DebugText[512];
			//HalDisplayString("RtlWriteRegistryValue\n");
			//_snprintf(DebugText, sizeof(DebugText), "RtlWriteRegistryValue %08x %08x %08x %08x\n", NtNameUs.Buffer, RegistryPath, RegistryPath->Buffer, RegistryPath->Length);
			//HalDisplayString(DebugText);
			// Write the registry device map, kbdclass expects to see this.
			// Do this only for the first created keyboard device.
			Status = RtlWriteRegistryValue(
				RTL_REGISTRY_DEVICEMAP,
				s_KeyboardPortKey,
				NtNameUs.Buffer,
				REG_SZ,
				RegistryPath->Buffer,
				RegistryPath->Length
			);
			if (NT_SUCCESS(Status)) s_CreatedOneKeyboardDevice = TRUE;
#if 0
			// RtlWriteRegistryValue is somehow bugchecking by recursive exception on NT 3.51;
			// so reimplement it here and see what's actually causing it.
			do {
				STRING RegistryPathStr;
				HalDisplayString("RtlInitString\n");
				RtlInitString(&RegistryPathStr, "\\Registry\\Machine\\Hardware\\DeviceMap\\KeyboardPort");
				UNICODE_STRING RegistryPathUs;
				HalDisplayString("RtlAnsiStringToUnicodeString\n");
				Status = RtlAnsiStringToUnicodeString(&RegistryPathUs, &RegistryPathStr, TRUE);
				
				OBJECT_ATTRIBUTES Oa;
				InitializeObjectAttributes(&Oa, &RegistryPathUs, OBJ_CASE_INSENSITIVE, NULL, NULL);
				HANDLE hKey;
				ULONG stackPtr;
				__asm__ __volatile__("mr %0,1" : "=r"(stackPtr));
				CHAR DebugText[512];
				_snprintf(DebugText, sizeof(DebugText), "NtCreateKey %08x\n", stackPtr);
				HalDisplayString(DebugText);
				Status = ZwCreateKey(&hKey, GENERIC_WRITE, &Oa, 0, NULL, 0, NULL);
				__asm__ __volatile__("mr %0,1" : "=r"(stackPtr));
				_snprintf(DebugText, sizeof(DebugText), "RtlFreeUnicodeString %08x\n", stackPtr);
				HalDisplayString(DebugText);
				RtlFreeUnicodeString(&RegistryPathUs);
				if (!NT_SUCCESS(Status)) {
					break;
				}
				__asm__ __volatile__("mr %0,1" : "=r"(stackPtr));
				// ZwSetValueKey dies
				// let's try NtSetValueKey.
				ULONG* SyscallTable = (ULONG*)(KeServiceDescriptorTable[0].Base);
				PAIXCALL_FPTR* FpZwSetValueKey = (PAIXCALL_FPTR*)ZwSetValueKey;
				PAIXCALL_FPTR* FpRtlFreeUnicodeString = (PAIXCALL_FPTR*)RtlFreeUnicodeString;
				AIXCALL_FPTR FpNtSetValueKey;
				FpNtSetValueKey.Function = (PVOID) (SyscallTable[(ULONG)FpZwSetValueKey[0]->Toc] & ~1);
				FpNtSetValueKey.Toc = FpRtlFreeUnicodeString[0]->Toc;
				// Memory barrier here, to prevent untoward over-optimisation
				asm volatile("" : : : "memory");
				typedef NTSTATUS (*tfpNtSetValueKey)(HANDLE KeyHandle, PUNICODE_STRING ValueName, ULONG TitleIndex, ULONG Type, PVOID Data, ULONG DataSize);
				tfpNtSetValueKey NtSetValueKey = (tfpNtSetValueKey)&FpNtSetValueKey;
				HalDisplayString("NtSetValueKey\n");
				Status = NtSetValueKey(hKey, &NtNameUs, 0, REG_SZ, RegistryPath->Buffer, RegistryPath->Length);
				HalDisplayString("NtClose\n");
				ZwClose(hKey);
				
				if (!NT_SUCCESS(Status)) break;
				s_CreatedOneKeyboardDevice = TRUE;
			} while (0);
#endif
#if 0
			CHAR DebugText[512];
			HalDisplayString("RtlWriteRegistryValue this\n");
			// Write the registry device map, kbdclass expects to see this.
			// Do this only for the first created keyboard device.
			Status = KbdWriteRegistryValue(
				NtNameUs.Buffer,
				RegistryPath->Buffer,
				RegistryPath->Length
			);
			if (NT_SUCCESS(Status)) s_CreatedOneKeyboardDevice = TRUE;
#endif
		}
		// Free the unicode string.
		//HalDisplayString("RtlFreeUnicodeString\n");
		RtlFreeUnicodeString(&NtNameUs);
		break;
	}
	
		//HalDisplayString("KbdInit out\n");
	return Status;
}