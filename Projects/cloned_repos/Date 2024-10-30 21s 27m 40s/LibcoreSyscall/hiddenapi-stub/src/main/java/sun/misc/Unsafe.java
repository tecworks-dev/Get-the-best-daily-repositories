/*
 * Copyright (C) 2007 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package sun.misc;

import java.lang.reflect.Field;

/**
 * The package name notwithstanding, this class is the quasi-standard
 * way for Java code to gain access to and use functionality which,
 * when unsupervised, would allow one to break the pointer/type safety
 * of Java.
 */
public final class Unsafe {

    /**
     * Traditional dalvik name.
     */
    private static final Unsafe THE_ONE = new Unsafe();
    /**
     * Traditional RI name.
     */
    private static final Unsafe theUnsafe = THE_ONE;

    /**
     * This class is only privately instantiable.
     */
    private Unsafe() {
    }

    /**
     * Gets the raw byte offset from the start of an object's memory to
     * the memory used to store the indicated instance field.
     *
     * @param field non-null; the field in question, which must be an instance field
     * @return the offset to the field
     */
    public long objectFieldOffset(Field field) {
        // Lsun/misc/Unsafe;->objectFieldOffset(Ljava/lang/reflect/Field;)J,core-platform-api,unsupported
        throw new AssertionError("stub");
    }

    /**
     * Gets a <code>long</code> field from the given object.
     *
     * @param obj    non-null; object containing the field
     * @param offset offset to the field within <code>obj</code>
     * @return the retrieved value
     */
    public native long getLong(Object obj, long offset);

    /**
     * Stores a <code>long</code> field into the given object.
     *
     * @param obj      non-null; object containing the field
     * @param offset   offset to the field within <code>obj</code>
     * @param newValue the value to store
     */
    public native void putLong(Object obj, long offset, long newValue);

    /**
     * Gets an <code>int</code> field from the given object.
     *
     * @param obj    non-null; object containing the field
     * @param offset offset to the field within <code>obj</code>
     * @return the retrieved value
     */
    public native int getInt(Object obj, long offset);

    /**
     * Stores an <code>int</code> field into the given object.
     *
     * @param obj      non-null; object containing the field
     * @param offset   offset to the field within <code>obj</code>
     * @param newValue the value to store
     */
    public native void putInt(Object obj, long offset, int newValue);

}
