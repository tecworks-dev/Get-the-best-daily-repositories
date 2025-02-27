
[![Visit our Discord server](https://discordapp.com/api/guilds/1301249042256363673/widget.png)](https://discord.gg/S8jWsc4SAq)

# Windows NT for GameCube/Wii

The following systems are supported:
* Nintendo GameCube
* Nintendo Wii
  * Wii Mini requires SD card hardmod (for now)
* Nintendo Wii U (**vWii only for now**)

The following systems are theoretically supported, although not tested due to the rarity of such hardware:
* Broadway Evaluation Board
* Cortado boards

The following systems will NEVER be supported:
* early Dolphin Development Hardware with only 4MB of usable RAM

## Drivers present

* Flipper interrupt controller (in HAL)
* Flipper Video Interface console framebuffer (YUV XFB) for ARC firmware and HAL
* Flipper GPU RGB framebuffer under NT (writing to EFB under text setup, writing to texture under GDI; copying out to XFB on vblank interrupt)
* Flipper Serial Interface (gamecube controller ports), supporting the following devices:
  * GameCube ASCII keyboard controller, plus unreleased English/European variants (discovered through reversing Phantasy Star Online); the latter are completely untested, the former has not been tested on real hardware
  * GameCube controller, with the following mappings:
    * Under ARC firmware: left analog stick and d-pad maps to up/down, A button maps to enter, B button maps to escape, X button maps to letter 'S'
    * Under NT text setup: left analog stick and d-pad maps to up/down, c-stick maps to page up/page down, A button maps to enter, B button maps to escape, X button maps to F8, Y button maps to letter 'C', Z button maps to letter 'L'
    * Under NT GDI: left analog stick moves mouse, A button maps to left mouse button, B button maps to right mouse button, L+R together maps to ctrl+alt+del, c-stick allows for choosing a keyboard scancode (1-9, 0, a-z), X button confirms the selected scancode. Numbers are first in the list so numeric-only text boxes (like entering CD key) still works.
  * N64 Randnet keyboard, completely untested so may have issues
  * N64 mouse (under NT only), completely untested so may have issues
  * N64 controller (completely untested so may have issues), with the following mappings:
    * Under ARC firmware: left analog stick and d-pad maps to up/down, A button maps to enter, B button maps to escape, Z button maps to letter 'S'
    * Under NT text setup: left analog stick and d-pad maps to up/down, c-stick maps to page up/page down, A button maps to enter, B button maps to escape, Z button maps to F8, L trigger maps to letter 'C', R trigger maps to letter 'L'
    * Under NT GDI: left analog stick moves mouse, A button maps to left mouse button, B button maps to right mouse button, L+R together maps to ctrl+alt+del, c-down and c-up allows for choosing a keyboard scancode (1-9, 0, a-z), start button confirms the selected scancode. Numbers are first in the list so numeric-only text boxes (like entering CD key) still works.
* Flipper External Interface (SPI bus), supporting the following devices:
  * RTC
  * USB Gecko (for kernel debugger only)
  * SD Gecko or compatible
  * IDE-EXI or compatible (has not been tested on real hardware)
* Vegas IOP IPC
* Vegas SDMC controller (via IOS)
* Vegas USB (OHCI/EHCI) controllers (via IOS), supporting the following devices:
   * USB keyboard
   * USB mouse
   * USB mass storage (currently has some issues, some devices may not work) 
   * **Hotplugging USB devices is not supported. To use a USB device, it must be plugged in before launching the ARC firmware.**

## Software compatibility

NT 3.51 RTM and higher. NT 3.51 betas (build 944 and below) will need kernel patches to run due to processor detection bugs. NT 3.5 will never be compatible, as it only supports PowerPC 601.
(The additional suspend/hibernation features in NT 3.51 PMZ could be made compatible in theory but in practise would require all of the additional drivers for that to be reimplemented.)

## Installing

### Preliminary

* Grab binaries from the release page, extract to SD card (or EXI-IDE device)
* Copy an NT 3.51 or 4.0 ISO to `sd:\nt\disk00.iso`
* Create a raw disk image of the size you want at `sd:\nt\disk00.img` - I use `qemu-img create disk00.img 2G`, change the size as appropriate. Remember that the maximum file size on a FAT32 partition is 4GB. 
* On a GameCube, load `arcldr_dol.dol` from Swiss; on Wii/vWii, load `arcldr` from the Homebrew Channel.

### Partitioning Disk

* When you get to ARC firmware menu, go to `Run firmware setup`, then `Repartition disk or disk image for NT installation`.
* Select the disk image you created earlier.
* Confirm the partition operation with Y (on keyboard), X button (on GameCube controller), or Z button (on N64 controller)
* When finished, the partitioner will ask to `Press any key to restart`. This should either restart your system or return to loader where you can load `arcldr` again.

### Installing NT

* Choose `Run NT setup from cd00`.
	* You will receive the message `Setup could not determine the type of computer you have`.
	* Choose `Other` (default selected option), just press `Enter` (or A button) when asked for hardware support disk.
	* Choose the HAL from the list, currently there is only one option: `Nintendo GameCube, Wii and Wii U (vWii)`.
* Next you will receive the message `Setup could not determine the type of one or more mass storage drivers installed in your system`. At least two drivers need to be loaded at this point.
	* To load a driver, press `S` (X button on GameCube controller, Z button on N64 controller) to pick a driver, choose `Other` from the list, press `Enter` (A button) when asked for hardware support disk, and choose the driver.
	  * `Nintendo Wii SD Slot (via IOS) [Disk Images]` is required when using the front SD card slot on a Wii or Wii U
	  * `Nintendo Wii USB (via IOS)` is required when using any USB device (keyboard, mouse or mass storage) on a Wii or Wii U
	  * `Nintendo GameCube Controller Ports` is required when using devices plugged into the GameCube controller ports on a GameCube or Wii
	  * `SD Gecko or IDE-EXI and Compatible [Disk Images]` is required when using SD Gecko (or compatible) or IDE-EXI (or compatible) devices in the GameCube memory card slots on a GameCube or Wii, or the serial ports present underneath a GameCube
   * To make this simpler: on a GameCube you will need only the last two; on a Wii U vWii you will only need the first two, and on a Wii you will need the first two and possibly the last two depending on if you are using/want to use the GameCube controller ports/memory card slots or not.
* You will receive the message `Setup could not determine the type of video adapter installed in the system`. Choose `Other` from the list, press `Enter` when asked for hardware support disk, and choose the correct option depending on the OS you are installing.
	* There are two options in this list; `ArtX Flipper, ATI Vegas, AMD Bollywood (NT 4)` is for NT 4, `ArtX Flipper, ATI Vegas, AMD Bollywood (NT 3.x)` is for NT 3.51.
* NT will boot and text setup will start. Go through the text setup.
* Under `Setup has determined that your computer contains the following hardware and software components`, change `Keyboard` from `Unknown` to `XT, AT or Enhanced Keyboard (83-104 keys)` and `Pointing Device` from `Unknown` to `No Mouse or Other Pointing Device`.
* Choose the `C:` drive from the partition list. If you chose to create an NT partition of size 2GB or less, it must be formatted.
* If you chose to create an NT partition of over 2GB in size, errors will be found by the disk examination process which will require a reboot. You will need to boot back into the ARC firmware from Swiss or the Homebrew Channel and follow the "Installing NT" steps again to get back to this point.
	* On the second attempt, disk examination will succeed, so just choose the `C:` partition again in the NT text setup partition selector.
* Proceed through the rest of NT text and graphical setup as normal.

## Known issues

* System may hang on reboot sometimes.
* There are issues with some USB mass storage devices.
* GDI driver uses slow unoptimised code for copying from GDI bitmap buffer to GPU texture buffer.
* ARC firmware and NT drivers support exFAT for disk images on an SD card/EXI-IDE device, but the loader currently does not support exFAT for loading the ARC firmware proper.
* The loader currently does not support loading the ARC firmware from a USB mass storage device.
* Be aware that the EXI bus is slower compared to other disk interfaces, so using SD Gecko/EXI-IDE causes slowdowns. This is most notable when installing NT on GameCube where this is the only available option.

## Building ARC firmware

You need devkitPPC. Additionally, a `libgcc.a` compiled for `powerpcle` must be present in `arcfw/gccle`. If you need to find one, it should be present on any Void Linux mirror, the current filename to search for as of 2024-07-12 is `cross-powerpcle-linux-gnu-0.34_1.x86_64.xbps` - decompress it by `zstdcat cross-powerpcle-linux-gnu-0.34_1.x86_64.xbps -o cross-powerpcle-linux-gnu-0.34_1.x86_64.tar`, then pull the file out of the tarball: `usr/lib/gcc/powerpcle-linux-gnu/10.2/libgcc.a`.

* Ensure `DEVKITPPC` environment variable is set to your devkitPPC directory, usually `/opt/devkitpro/devkitPPC`
* Build the ARC firmware loader: `cd arcldr ; make -f Makefile.rvl ; make -f Makefile.dol ; cd ..`
* Build the little endian libc: `cd arcfw/baselibc ; make ; cd ../..`
* Build the ARC firmware itself: `cd arcfw; make ; cd ..`

## Building HAL/drivers

You need [peppc](https://github.com/Wack0/peppc). Additionally, the powerpc libs from the [NT4 DDK](https://archive.org/details/94396011997WindowsNTDDKForWinNT4.0WorkstationUS.iso.7z) (`ddk/lib/ppc/free/*.lib`) must be present in `lib`. The rest of the toolchain (VC6 PPC CE cross compiler used for the C preprocessor for asm, as multi-line defines are handled improperly by gcc cpp; assembler PASM.EXE with single branch patched to skip "dump statements"; resource compiler and linker from MSVC 4.2, and its dependencies; `SPLITSYM.EXE` from NT 3.51 DDK to split COFF debug symbols from executables) is present in `msvc-ppc` directory.

The headers are included and come from various places with slight modifications for working with this toolchain, or for backwards compatibility reasons:
* `nt4/sdk` - NT4 SDK
* `nt4/ddk` - NT4 DDK (including all the headers from the `src/*/inc` directories)
* `nt4/crt` - VC++ 4.0 (CRT headers)
* `nt4/hal` - because of a lack of a public dump, this folder includes the headers that have evidence suggesting they were included in the NT4 halkit (minus `nthal.h` which is in the hal source folder, and was modified to allow for backwards compatibility). Some have been modified to allow them to be included by drivers after `ntddk.h` (so drivers can call `HalDisplayString` for debugging purposes, or use `LOADER_PARAMETER_BLOCK` to determine whether they are running in text setup or not).

The makefiles used are derived from devkitPro.

Ensure `PEPPC` environment variable is set to the `peppc-build/toolchain/bin` directory.

You must build the hal first (`cd halartx; make; cd ..`) before you can build the other drivers, as the HAL implements the exported IOP IPC and EXI drivers (due to the HAL itself using them).

## Acknowledgements

* libc used is [baselibc](https://github.com/PetteriAimonen/Baselibc)
* ELF loader, arcfw makefile (and some cache invalidation functions) adapted from [The Homebrew Channel](https://github.com/fail0verflow/hbc)
* Other makefiles adapted from [devkitPro](https://github.com/devkitPro/devkitppc-rules)
* Some lowlevel powerpc stuff, and ARC firmware framebuffer console implementation and font, adapted from [libogc](https://github.com/devkitPro/libogc)
* EXI-IDE driver in ARC loader adapted from [Swiss](https://github.com/emukidid/swiss-gc/blob/master/cube/swiss/source/devices/fat/ata.c)
* IOS IPC driver in ARC firmware adapted from [The Homebrew Channel's reload stub](https://github.com/fail0verflow/hbc/blob/master/channel/channelapp/stub/ios.c)
* ISO9660 FS implementation inside ARC firmware is [lib9660](https://github.com/erincandescent/lib9660) with some modifications.
* FAT FS implementation inside ARC firmware is [Petit FatFs](http://elm-chan.org/fsw/ff/00index_p.html) with some modifications; additionally the full [FatFs](http://elm-chan.org/fsw/ff/) is used for reading the underlying disk images on FAT16/FAT32/exFAT partitions (in ARC and inside iossdmc.sys and fpexiblk.sys)
* GDI driver derived from NT4 DDK example `framebuf`.
* Various drivers adapted from those in libogc.
