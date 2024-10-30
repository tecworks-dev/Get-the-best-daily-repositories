# Libcore-Syscall

Libcore-Syscall is a Java library for Android that allows you to make any Linux system calls directly from Java code.

## Features

- Support Android 5.0 - 15
- Support any system calls (as long as they are permitted by the seccomp filter)
- Implemented in 100% pure Java 1.8
- No shared libraries (lib*.so) are shipped with the library
- No `System.loadLibrary` or `System.load` is used
- Small, no dependencies

## Usage

The library provides the following classes:

- MemoryAccess/MemoryAllocator: Allocate and read/write native memory.
- NativeAccess: Register JNI methods, or call native functions (such as `dlopen`, `dlsym`, etc.) directly.
- Syscall: Make any Linux system calls.

## Example

Here is an example of how to use the library. It calls the `uname` system call to get the system information.

See [TestMainActivity.java](demo-app/src/main/java/com/example/test/app/TestMainActivity.java) for the complete example.

<details>

```java
import dev.tmpfs.libcoresyscall.core.IAllocatedMemory;
import dev.tmpfs.libcoresyscall.core.MemoryAccess;
import dev.tmpfs.libcoresyscall.core.MemoryAllocator;
import dev.tmpfs.libcoresyscall.core.NativeHelper;
import dev.tmpfs.libcoresyscall.core.Syscall;

public String unameDemo() {
    StringBuilder sb = new StringBuilder();
    int __NR_uname;
    switch (NativeHelper.getCurrentRuntimeIsa()) {
        case NativeHelper.ISA_X86_64:
            __NR_uname = 63;
            break;
        case NativeHelper.ISA_ARM64:
            __NR_uname = 160;
            break;
        // add other architectures here ...
    }
    // The struct of utsname can be found in <sys/utsname.h> in the NDK.
    // ...
    int releaseOffset = 65 * 2;
    // ...
    int utsSize = 65 * 6;
    try (IAllocatedMemory uts = MemoryAllocator.allocate(utsSize, true)) {
        long utsAddress = uts.getAddress();
        Syscall.syscall(__NR_uname, utsAddress);
        // ...
        sb.append("release = ").append(MemoryAccess.peekCString(utsAddress + releaseOffset));
        // ...
        sb.append("\n");
    } catch (ErrnoException e) {
        sb.append("ErrnoException: ").append(e.getMessage());
    }
    return sb.toString();
}
```

</details>

## The Tricks

- The Android-specific `libcore.io.Memory` and the evil `sun.misc.Unsafe` are used to access the native memory.
- Anonymous executable pages are allocated using the `android.system.Os.mmap` method.
- Native methods are registered with direct access to the `art::ArtMethod::entry_point_from_jni_` field.

## Notice

- This library is not intended to be used in production code. You use it once and it may crash everywhere. It is only for a Proof of Concept.
- This library can only work on ART, not on OpenJDK HotSpot / OpenJ9 / GraalVM.
- The `execmem` SELinux permission is required to allocate anonymous executable memory. Fortunately, this permission is granted to all app domain processes.
- The `system_server` does not have the `execmem` permission. However, this is not true if you have a system-wide Xposed framework installed.

## Build

To build the library:

```shell
./gradlew :core-syscall:assembleDebug
```

To build the demo app:

```shell
./gradlew :demo-app:assembleDebug
```

## The Future

- A symbol resolver that can resolve symbols in loaded native libraries.
- Loading arbitrary shared libraries (lib*.so) with 100% pure Java code in memory without writing it to the disk.

## Credits

- [pine](https://github.com/canyie/pine)

## License

The library is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
