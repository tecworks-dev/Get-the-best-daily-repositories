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

public class TrampolineCreatorFactory {

    private TrampolineCreatorFactory() {
        throw new AssertionError("no instances");
    }

    private static final HashMap<Integer, ITrampolineCreator> CREATOR_MAP = new HashMap<>(7);

    static {
        // add all supported ISAs
        CREATOR_MAP.put(NativeHelper.ISA_ARM64, TrampolineCreator_Arm64.INSTANCE);
        CREATOR_MAP.put(NativeHelper.ISA_X86_64, TrampolineCreator_X86_64.INSTANCE);
        CREATOR_MAP.put(NativeHelper.ISA_RISCV64, TrampolineCreator_Riscv64.INSTANCE);
        CREATOR_MAP.put(NativeHelper.ISA_ARM, TrampolineCreator_Arm.INSTANCE);
        CREATOR_MAP.put(NativeHelper.ISA_X86, TrampolineCreator_X86.INSTANCE);
        CREATOR_MAP.put(NativeHelper.ISA_MIPS64, TrampolineCreator_Mips64el.INSTANCE);
        CREATOR_MAP.put(NativeHelper.ISA_MIPS, TrampolineCreator_Mipsel.INSTANCE);
    }

    public static ITrampolineCreator create() {
        return create(NativeHelper.getCurrentRuntimeIsa());
    }

    public static ITrampolineCreator create(int isa) {
        ITrampolineCreator creator = CREATOR_MAP.get(isa);
        if (creator == null) {
            throw new UnsupportedOperationException("Unsupported ISA: " + NativeHelper.getIsaName(isa));
        }
        return creator;
    }

}
