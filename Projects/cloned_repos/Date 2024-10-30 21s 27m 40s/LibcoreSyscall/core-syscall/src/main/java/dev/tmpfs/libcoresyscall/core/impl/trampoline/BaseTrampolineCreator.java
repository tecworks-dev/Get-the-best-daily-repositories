package dev.tmpfs.libcoresyscall.core.impl.trampoline;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

public abstract class BaseTrampolineCreator implements ITrampolineCreator {

    protected BaseTrampolineCreator() {
    }

    /**
     * Get the padding instruction. It is used to fill the space between the end of the trampoline.
     * Typically, it is a breakpoint instruction.
     */
    public abstract byte[] getPaddingInstruction();

    public abstract Map<Method, byte[]> getNativeMethods();

    @Override
    public TrampolineInfo generateTrampoline(int pageSize) {
        // 1. Collect all native methods
        Map<Method, byte[]> nativeMethods = getNativeMethods();
        byte[] paddingInstruction = getPaddingInstruction();
        int paddingInstructionSize = paddingInstruction.length;
        if (paddingInstructionSize != 1 && paddingInstructionSize != 2 && paddingInstructionSize != 4) {
            throw new IllegalArgumentException("Invalid padding instruction size: " + paddingInstructionSize);
        }
        // 2. Allocate memory for trampoline.
        byte[] trampoline = new byte[pageSize];
        // 3. Layout the trampoline. We assume method entry should be aligned to 16 bytes.
        //    And there is no less than an 16 bytes gap between two methods.
        HashMap<Method, Integer> methodOffsets = new HashMap<>();
        int offset = 0;
        for (Map.Entry<Method, byte[]> entry : nativeMethods.entrySet()) {
            Method method = entry.getKey();
            byte[] code = entry.getValue();
            int codeSize = code.length;
            if (offset + codeSize + paddingInstructionSize > pageSize) {
                throw new IllegalStateException("the single page is not enough for all trampolines, this should not happen");
            }
            // Copy the code to trampoline
            System.arraycopy(code, 0, trampoline, offset, codeSize);
            methodOffsets.put(method, offset);
            offset += codeSize;
            // Fill the padding instruction
            int gapBytes = offset % 16;
            if (gapBytes == 0) {
                gapBytes += 16;
            }
            if (gapBytes % paddingInstructionSize != 0) {
                throw new IllegalStateException("padding instruction size is not aligned to gap bytes");
            }
            int paddingCount = gapBytes / paddingInstructionSize;
            for (int i = 0; i < paddingCount; i++) {
                System.arraycopy(paddingInstruction, 0, trampoline, offset, paddingInstructionSize);
                offset += paddingInstructionSize;
            }
        }
        // 4. Fill the padding instruction at the end of the trampoline
        int gapSize = pageSize - offset;
        int paddingCount = gapSize / paddingInstructionSize;
        for (int i = 0; i < paddingCount; i++) {
            System.arraycopy(paddingInstruction, 0, trampoline, offset, paddingInstructionSize);
            offset += paddingInstructionSize;
        }
        return new TrampolineInfo(trampoline, methodOffsets);
    }

}
