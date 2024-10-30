package dev.tmpfs.libcoresyscall.core.impl.trampoline;

public interface ITrampolineCreator {

    TrampolineInfo generateTrampoline(int pageSize);

}
