#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include "arc.h"
#include "runtime.h"
#include "arcconfig.h"
#include "arcdisk.h"
#include "arcio.h"
#include "arcenv.h"
#include "arcmem.h"
#include "arctime.h"
#include "arcconsole.h"
#include "arcfs.h"
#include "getstr.h"
#include "fatfs/ff.h"
#include "fatfs/diskio.h"

enum {
	SETUP_MENU_CHOICE_SYSPART,
	SETUP_MENU_CHOICE_IMGPART,
	SETUP_MENU_CHOICE_EXIT,
	SETUP_MENU_CHOICES_COUNT
};

enum {
	PARTITION_HD_MENU_CHOICES_COUNT = 16,
	PARTITION_HD_MENU_CHOICES_VISIBLE = 8
};

enum {
	PARTITION_MENU_CHOICE_CREATE,
	PARTITION_MENU_CHOICE_FINISH,
	PARTITION_MENU_CHOICE_CANCEL,
	PARTITION_MENU_CHOICES_COUNT
};

typedef char STRING_MENU_CHOICE [80];

static ULONG s_AdditionalDrives[100 - 1] = { 0 };

ULONG ArcDiskGetDiskType(const char* DevicePath);

static void PartitionerPrintDiskEntry(ULONG Hd, ULONG ExitIndex) {
	static const char* s_DiskNames[] = {
		"Memory Card Slot A",
		"Memory Card Slot B",
		"Serial Port 1",
		"Serial Port 2",
		"Front SD Slot",
		"USB Mass Storage",
		"Unknown"
	};
	if (Hd == ExitIndex) printf("Cancel");
	else {
		char HdVar[6];
		snprintf(HdVar, sizeof(HdVar), "hd%02d:", Hd);
		PCHAR HdDevice = ArcEnvGetDevice(HdVar);
		ULONG Drive = ArcDiskGetDiskType(HdDevice);
		if (Drive == 0xFFFFFFFF) Drive = (sizeof(s_DiskNames) / sizeof(s_DiskNames[0])) - 1;
		printf("%s (%dMB) - %s (%s)", HdVar, ArcDiskGetSizeMb(Hd), HdDevice, s_DiskNames[Drive]);
	}
}

static bool StringIsDigits(const char* str) {
	while (*str != 0) {
		if (*str < '0') return false;
		if (*str > '9') return false;
		str++;
	}
	return true;
}


ARC_STATUS ImgFfsToArc(FRESULT result);

static ARC_STATUS ArcFwImageCreate(ULONG HdIndex, ULONG ImgIndex, ULONG ImgSizeMb) {
	ULONG EmptySector[REPART_SECTOR_SIZE / sizeof(ULONG)];
	FIL f;
	char image_path[] = "0:/nt/disk00.img";
	image_path[0] = '0' + HdIndex;
	image_path[sizeof("0:/nt/disk") - 1 + 0] = '0' + (ImgIndex / 10);
	image_path[sizeof("0:/nt/disk") - 1 + 1] = '0' + (ImgIndex % 10);

	if (ImgSizeMb == 0) {
		// if zero length was passed, just attempt to delete
		f_unlink(image_path);
		return _ESUCCESS;
	}

	// memzero32 the data
	for (ULONG i = 0; i < sizeof(EmptySector) / sizeof(EmptySector[0]); i++) {
		EmptySector[i] = 0;
	}

	void ImgInvalidateLinkMap(UCHAR DeviceIndex, UCHAR DiskIndex);
	ImgInvalidateLinkMap(HdIndex, ImgIndex);

	// Open file as overwrite, expand to correct length, write zeroes to first sector, close.
	const char* operation = "open";
	memset(&f, 0, sizeof(f));
	FRESULT fr = f_open(&f, image_path, FA_READ | FA_WRITE | FA_CREATE_ALWAYS);
	ARC_STATUS Status = ImgFfsToArc(fr);
	if (ARC_SUCCESS(Status)) {
		do {
			operation = "expand";
			FSIZE_t NewSize = ((FSIZE_t)ImgSizeMb) * 0x100000;
			fr = f_lseek(&f, NewSize);
			Status = ImgFfsToArc(fr);
			if (ARC_FAIL(Status)) break;

			if (f.fptr != NewSize) {
				Status = _EIO;
				break;
			}
			fr = f_lseek(&f, 0);
			Status = ImgFfsToArc(fr);
			if (ARC_FAIL(Status)) break;

			operation = "write";
			ULONG wrote = 0;
			fr = f_write(&f, EmptySector, sizeof(EmptySector), &wrote);
			Status = ImgFfsToArc(fr);
			if (ARC_FAIL(Status)) break;
			if (wrote != sizeof(EmptySector)) Status = _EIO;
		} while (0);

		f_close(&f);
	}

	//if (ARC_FAIL(Status)) printf("%s failed: %d\r\n", operation, fr);

	return Status;
}

static void ArcFwImageCreatorSelected(const char* Name, ULONG Index, ULONG DiskSizeMb) {
	// Clear the screen.
	ArcSetScreenColour(ArcColourWhite, ArcColourBlue);
	ArcSetScreenAttributes(true, false, false);
	ArcClearScreen();
	ArcSetPosition(3, 0);

	printf("Free disk space on %s: %dMB\r\n", Name, DiskSizeMb);
	// 33 MB at the start of the disk is reserved (1MB for bootloader, partition table, etc; 32MB for arc system partition)
	// We also need at least 110MB of disk space for NT (95MB for NT 3.5x)
	// Can't install on any disk below that size.
	// Round it up to a clean 153MB (33MB + 120MB)
	if (DiskSizeMb < 153) {
		printf("At least 153MB of disk space is required for installation.\r\n");
		printf(" Press any key to continue...\r\n");
		IOSKBD_ReadChar();
		return;
	}

	ULONG RemainingDiskSize = DiskSizeMb - 33;
	printf("Available disk space on %s: %dMB\r\n", Name, RemainingDiskSize);

	// Get the partition size
	ULONG NtPartitionSize = 0;
	char TempName[ARC_ENV_MAXIMUM_VALUE_SIZE];
	while (true) {
		ArcClearScreen();
		ArcSetPosition(3, 0);
		printf("Free disk space on %s: %dMB\r\n", Name, DiskSizeMb);
		printf("Available disk space on %s: %dMB\r\n", Name, RemainingDiskSize);
		printf("Maximum size of NT operating system partition: %dMB\r\n", (RemainingDiskSize < REPART_MAX_NT_PART_IN_MB) ? RemainingDiskSize : REPART_MAX_NT_PART_IN_MB);

		static const char s_SizeText[] = "Enter size of NT operating system partition: ";
		ArcSetPosition(6, 5);
		printf("%s", "\x1B[2K");
		printf(s_SizeText);
		GETSTRING_ACTION Action;
		do {
			Action = KbdGetString(TempName, sizeof(TempName), NULL, 6, 5 + sizeof(s_SizeText) - 1);
		} while (Action != GetStringEscape && Action != GetStringSuccess);

		if (Action == GetStringEscape) {
			return;
		}

		if (!StringIsDigits(TempName)) continue;

		NtPartitionSize = (ULONG)atoll(TempName);
		if (NtPartitionSize < 120) {
			printf("\r\nAt least 120MB is required.\r\n");
			printf(" Press any key to continue...\r\n");
			IOSKBD_ReadChar();
			continue;
		}
		if (NtPartitionSize > REPART_MAX_NT_PART_IN_MB || NtPartitionSize > RemainingDiskSize) {
			printf("\r\nPartition size is too large.\r\n");
			printf(" Press any key to continue...\r\n");
			IOSKBD_ReadChar();
			continue;
		}

		break;
	}

	RemainingDiskSize -= NtPartitionSize;

	// Set up the menu for additional disk images.
	ULONG CountAdditional = 0;
	memset(s_AdditionalDrives, 0, sizeof(s_AdditionalDrives));

	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
	while (1) {
		ULONG DefaultChoice = 0;
		// Initialise the menu.
		PCHAR MenuChoices[PARTITION_MENU_CHOICES_COUNT] = {
			"Add additional disk image",
			"Finish and create images",
			"Cancel without making changes"
		};

		// Display the menu.
		ArcSetScreenColour(ArcColourWhite, ArcColourBlue);
		ArcSetScreenAttributes(true, false, false);
		ArcClearScreen();
		ArcSetPosition(3, 0);
		printf("Total disk space on %s: %dMB\r\n", Name, DiskSizeMb);
		printf("Remaining disk space on %s: %dMB\r\n", Name, RemainingDiskSize);

		for (ULONG i = 0; i < PARTITION_MENU_CHOICES_COUNT; i++) {
			ArcSetPosition(i + 5, 5);

			if (i == DefaultChoice) ArcSetScreenAttributes(true, false, true);

			printf("%s", MenuChoices[i]);
			ArcSetScreenAttributes(true, false, false);
		}

		ArcSetPosition(PARTITION_MENU_CHOICES_COUNT + 5, 0);

		printf("\r\nCurrent image list:\r\n");
		printf("NT operating system image: %dMB\r\n", NtPartitionSize + 33);
		for (ULONG i = 0; i < CountAdditional; i++) {
			printf("Additional image %d: %dMB\r\n", i + 1, s_AdditionalDrives[i]);
		}

		printf("\r\n Use the arrow keys to select.\r\n");
		printf(" Press Enter to choose.\r\n");
		printf("\r\n");

		// Implement the menu UI.
		UCHAR Character = 0;
		do {
			if (IOSKBD_CharAvailable()) {
				Character = IOSKBD_ReadChar();
				switch (Character) {
				case 0x1b:
					Character = IOSKBD_ReadChar();
					if (Character != '[') break;
					// fall-through: \x1b[ == \x9b
				case 0x9b:
					Character = IOSKBD_ReadChar();
					ArcSetPosition(DefaultChoice + 5, 5);
					printf("%s", MenuChoices[DefaultChoice]);

					switch (Character) {
					case 'A': // Up arrow
					case 'D': // Left arrow
						if (DefaultChoice == 0) DefaultChoice = PARTITION_MENU_CHOICES_COUNT;
						DefaultChoice--;
						break;
					case 'B': // Down arrow
					case 'C': // Right arrow
						DefaultChoice++;
						if (DefaultChoice >= PARTITION_MENU_CHOICES_COUNT) DefaultChoice = 0;
						break;
					case 'H': // Home
						DefaultChoice = 0;
						break;

					default:
						break;
					}

					ArcSetPosition(DefaultChoice + 5, 5);
					ArcSetScreenAttributes(true, false, true);
					printf("%s", MenuChoices[DefaultChoice]);
					ArcSetScreenAttributes(true, false, false);
					continue;

				default:
					break;
				}
			}
		} while ((Character != '\n') && (Character != '\r'));


		ArcClearScreen();
		ArcSetPosition(3, 0);

		if (DefaultChoice == PARTITION_MENU_CHOICE_CANCEL) return;

		if (DefaultChoice == PARTITION_MENU_CHOICE_FINISH) {
			printf("WARNING: EXISTING DISK IMAGES ON %s WILL BE DELETED!\r\n", Name);
			printf("Images to be created:\r\n");
			printf("NT operating system image: %dMB\r\n", NtPartitionSize + 33);
			for (ULONG i = 0; i < CountAdditional; i++) {
				printf("Additional image %d: %dMB\r\n", i + 1, s_AdditionalDrives[i]);
			}
			printf("PROCEED WITH OPERATION? (Y/N)\r\n");

			char Chr;
			do {
				Chr = IOSKBD_ReadChar();
			} while (Chr != 'Y' && Chr != 'y' && Chr != 'N' && Chr != 'n');

			if (Chr == 'N' || Chr == 'n') {
				printf("Operation cancelled.\r\n");
				printf(" Press any key to continue...\r\n");
				IOSKBD_ReadChar();
				return;
			}

			// create first image
			ULONG Image = 1;
			printf("Creating NT operating system image (image %d - this can be slow!)...\r\n", Image);
			ARC_STATUS Status = ArcFwImageCreate(Index, 0, NtPartitionSize + 33);

			const char* pArcPath = "multi(1)disk(0)rdisk(0)";
			char ArcPath[ARC_ENV_MAXIMUM_VALUE_SIZE];
			if (ARC_SUCCESS(Status)) {
				// partition first image
				if (Index < DEV_IOS_SDMC) {
					snprintf(ArcPath, sizeof(ArcPath), "multi(0)disk(%d)rdisk(0)", Index);
					pArcPath = ArcPath;
				}
				bool DataWritten = false;
				U32LE FileId;
				printf("Partitioning NT operating system image (image %d)...\r\n", Image);
				//printf("Open...\r\n");
				Status = Api->OpenRoutine(pArcPath, ArcOpenReadWrite, &FileId);
				if (ARC_SUCCESS(Status)) {
					//printf("Partition...\r\n");
					Status = ArcFsRepartitionDisk(FileId.v, NtPartitionSize, &DataWritten);
					Api->CloseRoutine(FileId.v);
				}
			}

			if (ARC_SUCCESS(Status)) {
				Image++;
				// create all other images
				for (ULONG i = 0; i < CountAdditional; i++, Image++) {
					printf("Creating additional image %d (image %d - this can be slow!)...\r\n", i + 1, Image);
					Status = ArcFwImageCreate(Index, i + 1, s_AdditionalDrives[i]);
					if (ARC_FAIL(Status)) break;
				}
			}

			if (ARC_SUCCESS(Status)) {
				// attempt to delete image
				ArcFwImageCreate(Index, CountAdditional + 1, 0);
			}

			if (ARC_FAIL(Status)) {
				printf("Failed to create image %d: %s\r\n", Image, ArcGetErrorString(Status));
				printf(" Press any key to continue...\r\n");
				IOSKBD_ReadChar();
				return;
			}

			printf("Created all disk images on %s successfully.\r\n", Name);
			// Specify that the drive we just partitioned is to be used for ARC NV storage.
			ArcEnvSetDeviceAfterFormat(Index);
			// Set the ARC system partition
			snprintf(TempName, sizeof(TempName), "%spartition(3)", pArcPath);
			Status = Api->SetEnvironmentRoutine("SYSTEMPARTITION", TempName);
			if (ARC_FAIL(Status)) printf("Could not set ARC system partition variable: %s\r\n", ArcGetErrorString(Status));
			printf(" Press any key to restart...\r\n");
			IOSKBD_ReadChar();
			Api->RestartRoutine();
			return;
		}

		if (DefaultChoice != PARTITION_MENU_CHOICE_CREATE) continue; // ???

		printf("Total disk space on %s: %dMB\r\n", Name, DiskSizeMb);
		printf("Remaining disk space on %s: %dMB\r\n", Name, RemainingDiskSize);

		while (true) {
			static const char s_SizeText[] = "Enter size of additional image: ";
			ArcSetPosition(5, 5);
			printf("%s", "\x1B[2K");
			printf(s_SizeText);
			GETSTRING_ACTION Action;
			do {
				Action = KbdGetString(TempName, sizeof(TempName), NULL, 5, 5 + sizeof(s_SizeText) - 1);
			} while (Action != GetStringEscape && Action != GetStringSuccess);

			if (Action == GetStringEscape) {
				return;
			}

			if (!StringIsDigits(TempName)) continue;

			s_AdditionalDrives[CountAdditional] = (ULONG)atoll(TempName);
			if (s_AdditionalDrives[CountAdditional] < 120) {
				printf("\r\nAt least 120MB is required.\r\n");
				printf(" Press any key to continue...\r\n");
				IOSKBD_ReadChar();
				continue;
			}


			if (s_AdditionalDrives[CountAdditional] > RemainingDiskSize) {
				printf("\r\nPartition size is too large.\r\n");
				printf(" Press any key to continue...\r\n");
				IOSKBD_ReadChar();
				continue;
			}

			RemainingDiskSize -= s_AdditionalDrives[CountAdditional];
			CountAdditional++;
			break;
		}
	}
	return;
}

static void ArcFwImageCreator(void) {
	static const char* s_DiskNames[] = {
		"Memory Card Slot A",
		"Memory Card Slot B",
		"Serial Port 1",
		"Serial Port 2",
		"Front SD Slot",
		"Cancel"
	};

	const char* DiskMenu[sizeof(s_DiskNames) / sizeof(s_DiskNames[0])];
	ULONG DiskIndexes[sizeof(s_DiskNames) / sizeof(s_DiskNames[0])];
	ULONG DiskSizes[sizeof(s_DiskNames) / sizeof(s_DiskNames[0])];

	ULONG MenuCount = 0;
	for (ULONG i = 0; i <= DEV_IOS_SDMC; i++) {
		if ((disk_status(i) & STA_NOINIT) != 0) continue;
		char path[3] = { '0', ':', 0 };
		path[0] = '0' + i;
		DWORD ClusterCount;
		FATFS* fsObj;
		if (f_getfree(path, &ClusterCount, &fsObj) != FR_OK) continue;
		// fsObj->csize must be power of 2.
		ULONG clusterShift = 31 - __builtin_clz(fsObj->csize);
		ULONG FreeSpaceMb = (((uint64_t)ClusterCount << clusterShift) / REPART_MB_SECTORS);
		if (FreeSpaceMb < 153) continue;

		DiskMenu[MenuCount] = s_DiskNames[i];
		DiskIndexes[MenuCount] = i;
		DiskSizes[MenuCount] = FreeSpaceMb;
		MenuCount++;
	}

	if (MenuCount == 0) {
		printf("No SD/EXI-SD/EXI-IDE device with at least 153MB of free space is present.\r\n");
		printf(" Press any key to continue...\r\n");
		IOSKBD_ReadChar();
		return;
	}

	DiskMenu[MenuCount] = s_DiskNames[(sizeof(s_DiskNames) / sizeof(s_DiskNames[0])) - 1];
	MenuCount++;

	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
	ULONG ExitIndex = MenuCount - 1;

	// First, select a disk.
	while (1) {
		ULONG DefaultChoice = 0;

		// Display the menu.
		ArcSetScreenColour(ArcColourWhite, ArcColourBlue);
		ArcSetScreenAttributes(true, false, false);
		ArcClearScreen();
		ArcSetPosition(3, 0);
		printf(" Select a disk:\r\n");

		for (ULONG i = 0; i < MenuCount; i++) {
			ArcSetPosition(i + 4, 5);

			if (i == DefaultChoice) ArcSetScreenAttributes(true, false, true);

			printf("%s", DiskMenu[i]);
			if (i != ExitIndex) {
				printf(" (%dMB free)", DiskSizes[i]);
			}
			ArcSetScreenAttributes(true, false, false);
		}

		ArcSetPosition(MenuCount + 4, 0);

		printf("\r\n Use the arrow keys to select.\r\n");
		printf(" Press Enter to choose.\r\n");
		printf("\n");

		// Implement the menu UI.
		UCHAR Character = 0;
		do {
			if (IOSKBD_CharAvailable()) {
				Character = IOSKBD_ReadChar();
				switch (Character) {
				case 0x1b:
					Character = IOSKBD_ReadChar();
					if (Character != '[') break;
					// fall-through: \x1b[ == \x9b
				case 0x9b:
					Character = IOSKBD_ReadChar();
					ArcSetPosition(DefaultChoice + 4, 5);
					printf("%s", DiskMenu[DefaultChoice]);
					if (DefaultChoice != ExitIndex) {
						printf(" (%dMB free)", DiskSizes[DefaultChoice]);
					}

					switch (Character) {
					case 'A': // Up arrow
					case 'D': // Left arrow
						if (DefaultChoice == 0) DefaultChoice = MenuCount;
						DefaultChoice--;
						break;
					case 'B': // Down arrow
					case 'C': // Right arrow
						DefaultChoice++;
						if (DefaultChoice >= MenuCount) DefaultChoice = 0;
						break;
					case 'H': // Home
						DefaultChoice = 0;
						break;

					default:
						break;
					}

					ArcSetPosition(DefaultChoice + 4, 5);
					ArcSetScreenAttributes(true, false, true);
					printf("%s", DiskMenu[DefaultChoice]);
					if (DefaultChoice != MenuCount) {
						printf(" (%dMB free)", DiskSizes[DefaultChoice]);
					}
					ArcSetScreenAttributes(true, false, false);
					continue;

				default:
					break;
				}
			}
		} while ((Character != '\n') && (Character != '\r'));

		// Clear the menu.
		for (ULONG i = 0; i < MenuCount; i++) {
			ArcSetPosition(i + 4, 5);
			printf("%s", "\x1B[2K");
		}

		// If cancel was selected, then return
		if (DefaultChoice == ExitIndex) return;

		ArcFwImageCreatorSelected(DiskMenu[DefaultChoice], DiskIndexes[DefaultChoice], DiskSizes[DefaultChoice]);
		return;
	}
}

static void ArcFwPartitionerSelected(ULONG Hd) {
	// Get number of USB disks
	ULONG HdCount;
	ArcDiskGetCounts(&HdCount, NULL);
	ULONG UsbCount = 0;
	for (ULONG i = 0; i < HdCount; i++) {
		if (!ArcEnvHardDiskIsUsb(i)) break;
		UsbCount++;
	}

	// Clear the screen.
	ArcSetScreenColour(ArcColourWhite, ArcColourBlue);
	ArcSetScreenAttributes(true, false, false);
	ArcClearScreen();
	ArcSetPosition(3, 0);

	// Get the disk size.
	ULONG DiskSizeMb = ArcDiskGetSizeMb(Hd);

	printf("Total disk space on hd%02d: %dMB\r\n", Hd, DiskSizeMb);
	// 33 MB at the start of the disk is reserved (1MB for bootloader, partition table, etc; 32MB for arc system partition)
	// We also need at least 110MB of disk space for NT (95MB for NT 3.5x)
	// Can't install on any disk below that size.
	// Round it up to a clean 153MB (33MB + 120MB)
	if (DiskSizeMb < 153) {
		printf("At least 153MB of disk space is required for installation.\r\n");
		printf(" Press any key to continue...\r\n");
		IOSKBD_ReadChar();
		return;
	}

	ULONG RemainingDiskSize = DiskSizeMb - 33;
	printf("Available disk space on hd%02d: %dMB\r\n", Hd, RemainingDiskSize);

	// If this is NOT a usb disk, we use the entire image size.
	ULONG NtPartitionSize = 0;
	char TempName[ARC_ENV_MAXIMUM_VALUE_SIZE];
	if (Hd >= UsbCount) {
		NtPartitionSize = RemainingDiskSize;
		if (NtPartitionSize > REPART_MAX_NT_PART_IN_MB) NtPartitionSize = REPART_MAX_NT_PART_IN_MB;
	}
	else {
		// Ask for the partition size.
		while (true) {
			ArcClearScreen();
			ArcSetPosition(3, 0);
			printf("Total disk space on hd%02d: %dMB\r\n", Hd, DiskSizeMb);
			printf("Available disk space on hd%02d: %dMB\r\n", Hd, RemainingDiskSize);
			printf("Maximum size of NT operating system partition: %dMB\r\n", (RemainingDiskSize < REPART_MAX_NT_PART_IN_MB) ? RemainingDiskSize : REPART_MAX_NT_PART_IN_MB);

			static const char s_SizeText[] = "Enter size of NT operating system partition: ";
			ArcSetPosition(6, 5);
			printf("%s", "\x1B[2K");
			printf(s_SizeText);
			GETSTRING_ACTION Action;
			do {
				Action = KbdGetString(TempName, sizeof(TempName), NULL, 6, 5 + sizeof(s_SizeText) - 1);
			} while (Action != GetStringEscape && Action != GetStringSuccess);

			if (Action == GetStringEscape) {
				return;
			}

			if (!StringIsDigits(TempName)) continue;

			NtPartitionSize = (ULONG)atoll(TempName);
			if (NtPartitionSize < 120) {
				printf("\r\nAt least 120MB is required.\r\n");
				printf(" Press any key to continue...\r\n");
				IOSKBD_ReadChar();
				continue;
			}
			if (NtPartitionSize > REPART_MAX_NT_PART_IN_MB || NtPartitionSize > RemainingDiskSize) {
				printf("\r\nPartition size is too large.\r\n");
				printf(" Press any key to continue...\r\n");
				IOSKBD_ReadChar();
				continue;
			}

			break;
		}
	}

	RemainingDiskSize -= NtPartitionSize;

	ArcClearScreen();
	ArcSetPosition(3, 0);
	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
	const char* DiskType = "DISK IMAGE";
	if (Hd < UsbCount) DiskType = "USB DISK";
	printf("WARNING: ALL DATA ON %s hd%02d: WILL BE LOST!\r\n", DiskType, Hd);
	printf("Partitions to be created:\r\n");
	printf("Reserved system space: 33MB\r\n");
	printf("NT operating system partition: %dMB\r\n", NtPartitionSize);
	printf("PROCEED WITH OPERATION? (Y/N)\r\n");
	// Allow S for YES and ESC for NO to accomodate gamecube/n64 controllers
	char Chr;
	do {
		Chr = IOSKBD_ReadChar();
	} while (Chr != 'Y' && Chr != 'y' && Chr != 'N' && Chr != 'n'&& Chr != 'S' && Chr != 's' && Chr != '\x1b');

	if (Chr == 'N' || Chr == 'n' || Chr == '\x1b') {
		printf("Operation cancelled.\r\n");
		printf(" Press any key to continue...\r\n");
		IOSKBD_ReadChar();
		return;
	}

	// open hd
	bool DataWritten = false;
	char HdVar[6];
	snprintf(HdVar, sizeof(HdVar), "hd%02d:", Hd);

	U32LE FileId;
	PCHAR HdDevice = ArcEnvGetDevice(HdVar);
	ARC_STATUS Status = Api->OpenRoutine(HdDevice, ArcOpenReadWrite, &FileId);
	if (ARC_SUCCESS(Status)) {
		Status = ArcFsRepartitionDisk(FileId.v, NtPartitionSize, &DataWritten);
		Api->CloseRoutine(FileId.v);
	}

	if (ARC_FAIL(Status)) {
		printf("Failed to repartition drive hd%02d: for NT: %s\r\n", Hd, ArcGetErrorString(Status));
		if (!DataWritten) printf("No data has been lost.\r\n");
	}
	else {
		printf("Repartitioned drive hd%02d: for NT successfully\r\n", Hd);
		// Specify that the drive we just partitioned is to be used for ARC NV storage.
		HdDevice = ArcEnvGetDevice(HdVar);
		// Depending on what kind of device this is, set the environment disk differently.
		ULONG Drive = ArcDiskGetDiskType(HdDevice);
		if (Drive > DEV_IOS_SDMC) ArcEnvSetDiskAfterFormat(HdDevice);
		else ArcEnvSetDeviceAfterFormat(Drive);
		// Set the ARC system partition
		snprintf(TempName, sizeof(TempName), "%spartition(3)", HdDevice);
		Status = Api->SetEnvironmentRoutine("SYSTEMPARTITION", TempName);
		if (ARC_FAIL(Status)) printf("Could not set ARC system partition environment variable: %s\r\n", ArcGetErrorString(Status));
		printf(" Press any key to restart...\r\n");
		IOSKBD_ReadChar();
		Api->RestartRoutine();
		return;
	}

	printf(" Press any key to continue...\r\n");
	IOSKBD_ReadChar();
	return;
}

static void ArcFwPartitioner(void) {
	ULONG HdCount;
	ArcDiskGetCounts(&HdCount, NULL);

	// First disks to be enumerated are USB. So this should work:
#if 0
	ULONG UsbCount = 0;
	for (ULONG i = 0; i < HdCount; i++) {
		if (!ArcEnvHardDiskIsUsb(i)) break;
		UsbCount++;
	}
#endif
	// Just show all disks here, at least for now.
	ULONG UsbCount = HdCount;

	ULONG MenuCount = UsbCount + 1;

	PVENDOR_VECTOR_TABLE Api = ARC_VENDOR_VECTORS();
	ULONG Rows = Api->GetDisplayStatusRoutine(0)->CursorMaxYPosition;
	ULONG MaxMenu = Rows - 7;
	if (MenuCount > MaxMenu) MenuCount = MaxMenu;
	ULONG ExitIndex = MenuCount - 1;
	
	// First, select a disk.
	while (1) {
		ULONG DefaultChoice = 0;

		// Display the menu.
		ArcSetScreenColour(ArcColourWhite, ArcColourBlue);
		ArcSetScreenAttributes(true, false, false);
		ArcClearScreen();
		ArcSetPosition(3, 0);
		printf(" Select a disk:\r\n");

		for (ULONG i = 0; i < MenuCount; i++) {
			ArcSetPosition(i + 4, 5);

			if (i == DefaultChoice) ArcSetScreenAttributes(true, false, true);

			PartitionerPrintDiskEntry(i, ExitIndex);
			ArcSetScreenAttributes(true, false, false);
		}

		ArcSetPosition(MenuCount + 4, 0);

		printf("\r\n Use the arrow keys to select.\r\n");
		printf(" Press Enter to choose.\r\n");
		printf("\n");

		// Implement the menu UI.
		UCHAR Character = 0;
		do {
			if (IOSKBD_CharAvailable()) {
				Character = IOSKBD_ReadChar();
				switch (Character) {
				case 0x1b:
					Character = IOSKBD_ReadChar();
					if (Character != '[') break;
					// fall-through: \x1b[ == \x9b
				case 0x9b:
					Character = IOSKBD_ReadChar();
					ArcSetPosition(DefaultChoice + 4, 5);
					PartitionerPrintDiskEntry(DefaultChoice, ExitIndex);

					switch (Character) {
					case 'A': // Up arrow
					case 'D': // Left arrow
						if (DefaultChoice == 0) DefaultChoice = HdCount;
						DefaultChoice--;
						break;
					case 'B': // Down arrow
					case 'C': // Right arrow
						DefaultChoice++;
						if (DefaultChoice >= MenuCount) DefaultChoice = 0;
						break;
					case 'H': // Home
						DefaultChoice = 0;
						break;

					default:
						break;
					}

					ArcSetPosition(DefaultChoice + 4, 5);
					ArcSetScreenAttributes(true, false, true);
					PartitionerPrintDiskEntry(DefaultChoice, ExitIndex);
					ArcSetScreenAttributes(true, false, false);
					continue;

				default:
					break;
				}
			}
		} while ((Character != '\n') && (Character != '\r'));

		// Clear the menu.
		for (ULONG i = 0; i < MenuCount; i++) {
			ArcSetPosition(i + 4, 5);
			printf("%s", "\x1B[2K");
		}

		// If cancel was selected, then return
		if (DefaultChoice == ExitIndex) return;

		ArcFwPartitionerSelected(DefaultChoice);
		return;
	}
}

static PCHAR ArcFwGetSystemPartitionDrive(PCHAR SysPart) {
	if (SysPart == NULL) {
		return NULL;
	}

	// get the partition
	PCHAR SysPartNumber = strstr(SysPart, "partition(");
	if (SysPartNumber == NULL) return NULL;
	SysPartNumber[0] = 0;
	return SysPart;
}

void ArcFwSetup(void) {
	while (1) {
		ULONG DefaultChoice = 0;
		// Initialise the menu.
		PCHAR MenuChoices[SETUP_MENU_CHOICES_COUNT] = {
			"Repartition disk or disk image for NT installation",
			"Create disk images on SD/IDE for NT installation (SLOW)",
			"Exit"
		};

		// Display the menu.
		ArcSetScreenColour(ArcColourWhite, ArcColourBlue);
		ArcSetScreenAttributes(true, false, false);
		ArcClearScreen();
		ArcSetPosition(3, 0);
		printf(" Firmware setup actions:\r\n");

		for (ULONG i = 0; i < SETUP_MENU_CHOICES_COUNT; i++) {
			ArcSetPosition(i + 4, 5);

			if (i == DefaultChoice) ArcSetScreenAttributes(true, false, true);

			printf("%s", MenuChoices[i]);
			ArcSetScreenAttributes(true, false, false);
		}

		ArcSetPosition(SETUP_MENU_CHOICES_COUNT + 4, 0);

		printf("\r\n Use the arrow keys to select.\r\n");
		printf(" Press Enter to choose.\r\n");
		printf("\n");

		// Implement the menu UI.
		UCHAR Character = 0;
		do {
			if (IOSKBD_CharAvailable()) {
				Character = IOSKBD_ReadChar();
				switch (Character) {
				case 0x1b:
					Character = IOSKBD_ReadChar();
					if (Character != '[') break;
					// fall-through: \x1b[ == \x9b
				case 0x9b:
					Character = IOSKBD_ReadChar();
					ArcSetPosition(DefaultChoice + 4, 5);
					printf("%s", MenuChoices[DefaultChoice]);

					switch (Character) {
					case 'A': // Up arrow
					case 'D': // Left arrow
						if (DefaultChoice == 0) DefaultChoice = SETUP_MENU_CHOICES_COUNT;
						DefaultChoice--;
						break;
					case 'B': // Down arrow
					case 'C': // Right arrow
						DefaultChoice++;
						if (DefaultChoice >= SETUP_MENU_CHOICES_COUNT) DefaultChoice = 0;
						break;
					case 'H': // Home
						DefaultChoice = 0;
						break;

					default:
						break;
					}

					ArcSetPosition(DefaultChoice + 4, 5);
					ArcSetScreenAttributes(true, false, true);
					printf("%s", MenuChoices[DefaultChoice]);
					ArcSetScreenAttributes(true, false, false);
					continue;

				default:
					break;
				}
			}
		} while ((Character != '\n') && (Character != '\r'));

		// Clear the menu.
		for (int i = 0; i < SETUP_MENU_CHOICES_COUNT; i++) {
			ArcSetPosition(i + 4, 5);
			printf("%s", "\x1B[2K");
		}

		// Execute the selected option.
		if (DefaultChoice == SETUP_MENU_CHOICE_EXIT) {
			return;
		}

		if (DefaultChoice == SETUP_MENU_CHOICE_IMGPART) {
			ArcFwImageCreator();
			return;
		}

		if (DefaultChoice == SETUP_MENU_CHOICE_SYSPART) {
			ArcFwPartitioner();
			return;
		}
	}
}