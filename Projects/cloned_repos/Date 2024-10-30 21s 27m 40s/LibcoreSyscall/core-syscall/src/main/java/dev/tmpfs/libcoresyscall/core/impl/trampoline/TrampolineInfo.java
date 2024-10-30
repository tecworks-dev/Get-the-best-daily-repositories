package dev.tmpfs.libcoresyscall.core.impl.trampoline;

import androidx.annotation.NonNull;

import java.lang.reflect.Method;
import java.util.Map;

public class TrampolineInfo {

    @NonNull
    public final byte[] trampolineCode;

    @NonNull
    public final Map<Method, Integer> nativeEntryOffsetMap;

    public TrampolineInfo(@NonNull byte[] trampolineCode, @NonNull Map<Method, Integer> nativeEntryOffsetMap) {
        this.trampolineCode = trampolineCode;
        this.nativeEntryOffsetMap = nativeEntryOffsetMap;
    }

}
