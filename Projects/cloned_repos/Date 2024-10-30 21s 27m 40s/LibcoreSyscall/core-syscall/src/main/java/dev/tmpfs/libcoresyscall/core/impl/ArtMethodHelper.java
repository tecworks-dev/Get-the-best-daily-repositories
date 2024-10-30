package dev.tmpfs.libcoresyscall.core.impl;

import android.annotation.SuppressLint;
import android.os.Build;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Field;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Objects;

import dev.tmpfs.libcoresyscall.core.NativeHelper;
import libcore.io.Memory;
import sun.misc.Unsafe;

public class ArtMethodHelper {

    private ArtMethodHelper() {
        throw new AssertionError("no instances");
    }

    private static Unsafe sUnsafe = null;

    private static Unsafe getUnsafe() {
        if (sUnsafe == null) {
            try {
                @SuppressLint("DiscouragedPrivateApi")
                Field field = Unsafe.class.getDeclaredField("theUnsafe");
                field.setAccessible(true);
                sUnsafe = (Unsafe) field.get(null);
            } catch (ReflectiveOperationException e) {
                throw ReflectHelper.unsafeThrow(e);
            }
        }
        return sUnsafe;
    }

    @RequiresApi(23)
    private static long getArtMethodFromReflectedMethodForApi23To25(@NonNull Member method) {
        try {
            Class<?> kAbstractMethod = Class.forName("java.lang.reflect.AbstractMethod");
            Field artMethod = kAbstractMethod.getDeclaredField("artMethod");
            artMethod.setAccessible(true);
            return (long) Objects.requireNonNull(artMethod.get(method));
        } catch (ReflectiveOperationException e) {
            throw ReflectHelper.unsafeThrow(e);
        }
    }

    @RequiresApi(26)
    private static long getArtMethodFromReflectedMethodAboveApi26(@NonNull Member method) {
        try {
            // Ljava/lang/reflect/Executable;->artMethod:J,unsupported
            //noinspection JavaReflectionMemberAccess
            Field artMethod = Executable.class.getDeclaredField("artMethod");
            artMethod.setAccessible(true);
            return (long) Objects.requireNonNull(artMethod.get(method));
        } catch (ReflectiveOperationException e) {
            throw ReflectHelper.unsafeThrow(e);
        }
    }

    /**
     * Get the ArtMethod from a reflected method or constructor.
     *
     * @param method method or constructor
     * @return the ArtMethod address
     */
    private static long getArtMethodAddressFromReflectedMethod(@NonNull Member method) {
        if (Build.VERSION.SDK_INT >= 26) {
            return getArtMethodFromReflectedMethodAboveApi26(method);
        } else if (Build.VERSION.SDK_INT >= 23) {
            return getArtMethodFromReflectedMethodForApi23To25(method);
        } else {
            throw new UnsupportedOperationException("unsupported API: " + Build.VERSION.SDK_INT);
        }
    }

    private static Object getArtMethodObjectBelowSdk23(@NonNull Member method) {
        try {
            Class<?> kArtMethod = Class.forName("java.lang.reflect.ArtMethod");
            Class<?> kAbstractMethod = Class.forName("java.lang.reflect.AbstractMethod");
            Field artMethod = kAbstractMethod.getDeclaredField("artMethod");
            artMethod.setAccessible(true);
            return kArtMethod.cast(artMethod.get(method));
        } catch (ReflectiveOperationException e) {
            throw ReflectHelper.unsafeThrow(e);
        }
    }

    private static long sArtMethodNativeEntryPointOffset = 0;

    private static long getArtMethodEntryPointFromJniOffset() {
        if (sArtMethodNativeEntryPointOffset != 0) {
            return sArtMethodNativeEntryPointOffset;
        }
        // For Android 6.0+/SDK23+, ArtMethod is no longer a mirror object.
        // We need to calculate the offset of the art::ArtMethod::entry_point_from_jni_ field.
        // See https://github.com/canyie/pine/blob/master/core/src/main/cpp/art/art_method.h
        boolean is64Bit = NativeHelper.isCurrentRuntime64Bit();
        switch (Build.VERSION.SDK_INT) {
            case Build.VERSION_CODES.LOLLIPOP:
                sArtMethodNativeEntryPointOffset = 32;
                break;
            case Build.VERSION_CODES.LOLLIPOP_MR1:
                sArtMethodNativeEntryPointOffset = is64Bit ? 48 : 40;
                break;
            case Build.VERSION_CODES.M:
                sArtMethodNativeEntryPointOffset = is64Bit ? 40 : 32;
                break;
            case Build.VERSION_CODES.N:
            case Build.VERSION_CODES.N_MR1:
                sArtMethodNativeEntryPointOffset = is64Bit ? 40 : 28;
                break;
            case Build.VERSION_CODES.O:
            case Build.VERSION_CODES.O_MR1:
                sArtMethodNativeEntryPointOffset = is64Bit ? 32 : 24;
                break;
            case Build.VERSION_CODES.P:
            case Build.VERSION_CODES.Q:
            case Build.VERSION_CODES.R:
                sArtMethodNativeEntryPointOffset = is64Bit ? 24 : 20;
                break;
            case Build.VERSION_CODES.S:
            case Build.VERSION_CODES.S_V2:
            case Build.VERSION_CODES.TIRAMISU:
            case Build.VERSION_CODES.UPSIDE_DOWN_CAKE:
            case Build.VERSION_CODES.VANILLA_ICE_CREAM:
                sArtMethodNativeEntryPointOffset = 16;
                break;
            default:
                // use last/latest known offset
                sArtMethodNativeEntryPointOffset = 16;
                break;
        }
        return sArtMethodNativeEntryPointOffset;
    }

    public static void registerNativeMethod(@NonNull Member method, long address) {
        if (!(method instanceof Method) && !(method instanceof Constructor)) {
            throw new IllegalArgumentException("method must be a method or constructor");
        }
        if (address == 0) {
            throw new IllegalArgumentException("address must not be 0");
        }
        int modifiers = method.getModifiers();
        if (!Modifier.isNative(modifiers)) {
            throw new IllegalArgumentException("method must be native: " + method);
        }
        if (!NativeHelper.isCurrentRuntime64Bit()) {
            // check overflow
            if ((address & 0xFFFFFFFF00000000L) != 0) {
                throw new IllegalArgumentException("address overflow in 32-bit mode: " + Long.toHexString(address));
            }
        }
        try {
            Class<?> declaringClass = method.getDeclaringClass();
            // JNI specification says that the class needs to be initialized before the native method is registered
            Class.forName(declaringClass.getName(), true, declaringClass.getClassLoader());
        } catch (ClassNotFoundException e) {
            // should not happen
            throw ReflectHelper.unsafeThrow(e);
        }
        if (Build.VERSION.SDK_INT < 23) {
            Object artMethod = getArtMethodObjectBelowSdk23(method);
            if (artMethod == null) {
                throw new IllegalArgumentException("unable to get ArtMethod from " + method);
            }
            long offset = getArtMethodEntryPointFromJniOffset();
            if (offset == 0) {
                throw new IllegalStateException("unable to get ArtMethod::entry_point_from_jni_ offset");
            }
            if (NativeHelper.isCurrentRuntime64Bit()) {
                getUnsafe().putLong(artMethod, offset, address);
            } else {
                getUnsafe().putInt(artMethod, offset, (int) address);
            }
        } else {
            // for API 23 and above
            long artMethod = getArtMethodAddressFromReflectedMethod(method);
            if (artMethod == 0) {
                throw new IllegalArgumentException("unable to get ArtMethod from " + method);
            }
            long offset = getArtMethodEntryPointFromJniOffset();
            if (offset == 0) {
                throw new IllegalStateException("unable to get ArtMethod::entry_point_from_jni_ offset");
            }
            long addr = artMethod + offset;
            // actual native method registration
            if (NativeHelper.isCurrentRuntime64Bit()) {
                Memory.pokeLong(addr, address, false);
            } else {
                Memory.pokeInt(addr, (int) address, false);
            }
        }
    }

    private static long getArtMethodEntryPointFromJniRaw(@NonNull Member method) {
        if (!(method instanceof Method) && !(method instanceof Constructor)) {
            throw new IllegalArgumentException("method must be a method or constructor");
        }
        int modifiers = method.getModifiers();
        if (!Modifier.isNative(modifiers)) {
            throw new IllegalArgumentException("method must be native: " + method);
        }
        if (Build.VERSION.SDK_INT < 23) {
            Object artMethod = getArtMethodObjectBelowSdk23(method);
            if (artMethod == null) {
                throw new IllegalArgumentException("unable to get ArtMethod from " + method);
            }
            long offset = getArtMethodEntryPointFromJniOffset();
            if (offset == 0) {
                throw new IllegalStateException("unable to get ArtMethod::entry_point_from_jni_ offset");
            }
            if (NativeHelper.isCurrentRuntime64Bit()) {
                return getUnsafe().getLong(artMethod, offset);
            } else {
                return ((long) getUnsafe().getInt(artMethod, offset)) & 0xFFFFFFFFL;
            }
        } else {
            // for API 23 and above
            long artMethod = getArtMethodAddressFromReflectedMethod(method);
            if (artMethod == 0) {
                throw new IllegalArgumentException("unable to get ArtMethod from " + method);
            }
            long offset = getArtMethodEntryPointFromJniOffset();
            if (offset == 0) {
                throw new IllegalStateException("unable to get ArtMethod::entry_point_from_jni_ offset");
            }
            long addr = artMethod + offset;
            // actual native method registration
            if (NativeHelper.isCurrentRuntime64Bit()) {
                return Memory.peekLong(addr, false);
            } else {
                return ((long) Memory.peekInt(addr, false)) & 0xFFFFFFFFL;
            }
        }
    }

    private static class NeverCall {

        private NeverCall() {
            throw new AssertionError("never call");
        }

        native void nativeNeverCall();

    }

    private static long sArtJniDlsymLookupStub = 0;

    public static long getJniDlsymLookupStub() {
        if (sArtJniDlsymLookupStub != 0) {
            return sArtJniDlsymLookupStub;
        }
        Method nativeNeverCall;
        try {
            nativeNeverCall = NeverCall.class.getDeclaredMethod("nativeNeverCall");
        } catch (NoSuchMethodException e) {
            // should not happen
            throw ReflectHelper.unsafeThrow(e);
        }
        long entryPoint = getArtMethodEntryPointFromJniRaw(nativeNeverCall);
        if (entryPoint == 0) {
            throw new IllegalStateException("unable to get ArtMethod::entry_point_from_jni_ for " + nativeNeverCall);
        }
        sArtJniDlsymLookupStub = entryPoint;
        return sArtJniDlsymLookupStub;
    }

    public static void unregisterNativeMethod(@NonNull Member method) {
        registerNativeMethod(method, getJniDlsymLookupStub());
    }

    public static long getRegisteredNativeMethod(@NonNull Member method) {
        long entryPoint = getArtMethodEntryPointFromJniRaw(method);
        if (entryPoint == getJniDlsymLookupStub()) {
            return 0;
        }
        return entryPoint;
    }

}
