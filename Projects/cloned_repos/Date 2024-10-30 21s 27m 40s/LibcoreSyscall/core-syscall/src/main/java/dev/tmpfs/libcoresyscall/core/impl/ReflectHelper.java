package dev.tmpfs.libcoresyscall.core.impl;

import java.lang.reflect.InvocationTargetException;

public class ReflectHelper {

    private ReflectHelper() {
    }

    @SuppressWarnings("unchecked")
    private static <T extends Throwable> AssertionError unsafeThrowImpl(Throwable t) throws T {
        throw (T) t;
    }

    public static AssertionError unsafeThrow(Throwable t) {
        return unsafeThrowImpl(t);
    }

    public static AssertionError unsafeThrowForIteCause(Throwable t) {
        unsafeThrowImpl(getIteCauseOrSelf(t));
        throw new AssertionError("unreachable");
    }

    public static Throwable getIteCauseOrSelf(Throwable t) {
        Throwable cause;
        if (t instanceof InvocationTargetException && (cause = t.getCause()) != null) {
            return cause;
        } else {
            return t;
        }
    }

}
