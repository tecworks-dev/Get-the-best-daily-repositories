package dev.tmpfs.libcoresyscall.core.impl;

import android.system.ErrnoException;
import android.system.Os;
import android.system.OsConstants;

import java.lang.reflect.Method;
import java.util.Map;

import dev.tmpfs.libcoresyscall.core.Syscall;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.CommonSyscallNumberTables;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.ISyscallNumberTable;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.ITrampolineCreator;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.TrampolineCreatorFactory;
import dev.tmpfs.libcoresyscall.core.impl.trampoline.TrampolineInfo;
import libcore.io.Memory;

public class NativeBridge {

    private NativeBridge() {
        throw new AssertionError("no instances");
    }

    private static long sPageSize = 0;
    private static long sTrampolineBase = 0;
    private static boolean sNativeMethodRegistered = false;
    private static boolean sTrampolineSetReadOnly = false;

    public static native long nativeSyscall(int number, long arg1, long arg2, long arg3, long arg4, long arg5, long arg6);

    public static native void nativeClearCache(long address, long size);

    public static native long nativeCallPointerFunction0(long function);

    public static native long nativeCallPointerFunction1(long function, long arg1);

    public static native long nativeCallPointerFunction2(long function, long arg1, long arg2);

    public static native long nativeCallPointerFunction3(long function, long arg1, long arg2, long arg3);

    public static native long nativeCallPointerFunction4(long function, long arg1, long arg2, long arg3, long arg4);

    public static native long nativeGetJavaVM();

    public static long getPageSize() {
        long ps = sPageSize;
        if (ps == 0) {
            ps = Os.sysconf(OsConstants._SC_PAGESIZE);
            if (ps != 4096 && ps != 16384 && ps != 65536) {
                throw new AssertionError("Unexpected page size: " + ps);
            }
            sPageSize = ps;
        }
        return ps;
    }

    public static synchronized void initializeOnce() {
        if (!sNativeMethodRegistered) {
            // 1. Get the page size.
            long pageSize = getPageSize();
            // 2. Prepare the trampoline.
            ITrampolineCreator creator = TrampolineCreatorFactory.create();
            TrampolineInfo trampoline = creator.generateTrampoline((int) pageSize);
            // 3. Allocate a memory region for the trampoline.
            final int MAP_ANONYMOUS = 0x20;
            long address;
            try {
                address = Os.mmap(0, pageSize,
                        OsConstants.PROT_READ | OsConstants.PROT_WRITE | OsConstants.PROT_EXEC,
                        OsConstants.MAP_PRIVATE | MAP_ANONYMOUS, null, 0);
            } catch (ErrnoException e) {
                if (e.errno == OsConstants.EACCES) {
                    throw new UnsupportedOperationException("mmap failed with EACCES. Please check SELinux policy for execmem permission.");
                }
                // I have no idea what happened.
                throw ReflectHelper.unsafeThrow(e);
            }
            // 4. Write the trampoline to the memory region.
            Memory.pokeByteArray(address, trampoline.trampolineCode, 0, trampoline.trampolineCode.length);
            // 5. Register the native method.
            for (Map.Entry<Method, Integer> methods : trampoline.nativeEntryOffsetMap.entrySet()) {
                Method method = methods.getKey();
                int offset = methods.getValue();
                long function = address + (long) offset;
                ArtMethodHelper.registerNativeMethod(method, function);
            }
            sTrampolineBase = address;
            sNativeMethodRegistered = true;
        }
        if (!sTrampolineSetReadOnly) {
            // 6. Clear the instruction cache.
            // Unfortunately, there is no way to clear the instruction cache without using JNI.
            // The I-cache is typically empty since we the page is just allocated.
            // But we had better flush the D-cache to make sure the CPU loads the latest instructions into the I-cache.
            // If we successfully invoke the nativeClearCache method without a crash, then there will be no further issues.
            // TODO: 2024-10-28 Do something to clear the d-cache.
            NativeBridge.nativeClearCache(sTrampolineBase, getPageSize());
            // 7. Set the memory region to read-only.
            long pageSize = getPageSize();
            ISyscallNumberTable sysnr = CommonSyscallNumberTables.get();
            try {
                long rc = NativeBridge.nativeSyscall(sysnr.__NR_mprotect(), sTrampolineBase, pageSize,
                        OsConstants.PROT_READ | OsConstants.PROT_EXEC, 0, 0, 0);
                Syscall.checkResultOrThrow(rc, "mprotect");
            } catch (ErrnoException e) {
                throw ReflectHelper.unsafeThrow(e);
            }
            sTrampolineSetReadOnly = true;
        }
        // Done.
    }
}
