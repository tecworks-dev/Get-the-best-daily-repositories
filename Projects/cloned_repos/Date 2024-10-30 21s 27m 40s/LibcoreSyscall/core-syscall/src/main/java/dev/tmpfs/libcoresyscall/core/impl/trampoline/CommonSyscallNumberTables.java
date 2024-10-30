package dev.tmpfs.libcoresyscall.core.impl.trampoline;

import java.util.HashMap;

import dev.tmpfs.libcoresyscall.core.NativeHelper;
import dev.tmpfs.libcoresyscall.core.impl.arch.TrampolineCreator_Arm;
import dev.tmpfs.libcoresyscall.core.impl.arch.TrampolineCreator_Arm64;
import dev.tmpfs.libcoresyscall.core.impl.arch.TrampolineCreator_Mips64el;
import dev.tmpfs.libcoresyscall.core.impl.arch.TrampolineCreator_Mipsel;
import dev.tmpfs.libcoresyscall.core.impl.arch.TrampolineCreator_Riscv64;
import dev.tmpfs.libcoresyscall.core.impl.arch.TrampolineCreator_X86;
import dev.tmpfs.libcoresyscall.core.impl.arch.TrampolineCreator_X86_64;

public class CommonSyscallNumberTables {

    private CommonSyscallNumberTables() {
        throw new AssertionError("no instances");
    }

    private static final HashMap<Integer, ISyscallNumberTable> SYSCALL_NUMBER_TABLE_MAP = new HashMap<>(7);

    static {
        // add all supported ISAs
        SYSCALL_NUMBER_TABLE_MAP.put(NativeHelper.ISA_ARM64, TrampolineCreator_Arm64.INSTANCE);
        SYSCALL_NUMBER_TABLE_MAP.put(NativeHelper.ISA_X86_64, TrampolineCreator_X86_64.INSTANCE);
        SYSCALL_NUMBER_TABLE_MAP.put(NativeHelper.ISA_RISCV64, TrampolineCreator_Riscv64.INSTANCE);
        SYSCALL_NUMBER_TABLE_MAP.put(NativeHelper.ISA_ARM, TrampolineCreator_Arm.INSTANCE);
        SYSCALL_NUMBER_TABLE_MAP.put(NativeHelper.ISA_X86, TrampolineCreator_X86.INSTANCE);
        SYSCALL_NUMBER_TABLE_MAP.put(NativeHelper.ISA_MIPS64, TrampolineCreator_Mips64el.INSTANCE);
        SYSCALL_NUMBER_TABLE_MAP.put(NativeHelper.ISA_MIPS, TrampolineCreator_Mipsel.INSTANCE);
    }

    public static ISyscallNumberTable get() {
        return get(NativeHelper.getCurrentRuntimeIsa());
    }

    public static ISyscallNumberTable get(int isa) {
        ISyscallNumberTable table = SYSCALL_NUMBER_TABLE_MAP.get(isa);
        if (table == null) {
            throw new UnsupportedOperationException("Unsupported ISA: " + NativeHelper.getIsaName(isa));
        }
        return table;
    }

}
