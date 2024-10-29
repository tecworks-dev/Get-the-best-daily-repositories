package android.content.res;

import android.graphics.Color;
import android.graphics.Movie;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.text.Html;
import android.util.DisplayMetrics;
import android.util.TypedValue;

import java.util.HashMap;

import de.robv.android.xposed.IXposedHookZygoteInit;
import de.robv.android.xposed.XposedBridge.CopyOnWriteSortedSet;
import xposed.dummy.XResourcesSuperClass;

import static de.robv.android.xposed.XposedHelpers.findAndHookMethod;

/**
 * {@link android.content.res.Resources} subclass that allows replacing individual resources.
 *
 * <p>Xposed replaces the standard resources with this class, which overrides the methods used for
 * retrieving individual resources and adds possibilities to replace them. These replacements can
 * be set using the methods made available via the API methods in this class.
 */
@SuppressWarnings("JniMissingFunction")
public class XResources extends XResourcesSuperClass {
	/** Dummy, will never be called (objects are transferred to this class only). */
	private XResources() {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	public void initObject(String resDir) {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	public boolean isFirstLoad() {
        throw new UnsupportedOperationException();
	}

	/** @hide */
	public static void setPackageNameForResDir(String packageName, String resDir) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Returns the name of the package that these resources belong to, or "android" for system resources.
	 */
	public String getPackageName() {
		throw new UnsupportedOperationException();
	}

	/**
	 * Special case of {@link #getPackageName} during object creation.
	 *
	 * <p>For a short moment during/after the creation of a new {@link android.content.res Resources}
	 * object, it isn't an instance of {@link XResources} yet. For any hooks that need information
	 * about the just created object during this particular stage, this method will return the
	 * package name.
	 *
	 * <p class="warning">If you call this method outside of {@code getTopLevelResources()}, it
	 * throws an {@code IllegalStateException}.
	 */
	public static String getPackageNameDuringConstruction() {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	public static void init(ThreadLocal<Object> latestResKey) throws Exception {
		throw new UnsupportedOperationException();
	}

	/**
	 * Wrapper for information about an indiviual resource.
	 */
	public static class ResourceNames {
		/** The resource ID. */
		public final int id;
		/** The resource package name as returned by {@link #getResourcePackageName}. */
		public final String pkg;
		/** The resource entry name as returned by {@link #getResourceEntryName}. */
		public final String name;
		/** The resource type name as returned by {@link #getResourceTypeName}. */
		public final String type;
		/** The full resource nameas returned by {@link #getResourceName}. */
		public final String fullName;

		private ResourceNames(int id, String pkg, String name, String type) {
			this.id = id;
			this.pkg = pkg;
			this.name = name;
			this.type = type;
			this.fullName = pkg + ":" + type + "/" + name;
		}

		/**
		 * Returns whether all non-null parameters match the values of this object.
		 */
		public boolean equals(String pkg, String name, String type, int id) {
			return (pkg  == null || pkg.equals(this.pkg))
				&& (name == null || name.equals(this.name))
				&& (type == null || type.equals(this.type))
				&& (id == 0 || id == this.id);
		}
	}

	// =======================================================
	//   DEFINING REPLACEMENTS
	// =======================================================

	/**
	 * Sets a replacement for an individual resource. See {@link #setReplacement(String, String, String, Object)}.
	 *
	 * @param id The ID of the resource which should be replaced.
	 * @param replacement The replacement, see above.
	 */
	public void setReplacement(int id, Object replacement) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Sets a replacement for an individual resource. See {@link #setReplacement(String, String, String, Object)}.
	 *
	 * @deprecated Use {@link #setReplacement(String, String, String, Object)} instead.
	 *
	 * @param fullName The full resource name, e.g. {@code com.example.myapplication:string/app_name}.
	 *                 See {@link #getResourceName}.
	 * @param replacement The replacement.
	 */
	@Deprecated
	public void setReplacement(String fullName, Object replacement) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Sets a replacement for an individual resource. If called more than once for the same ID, the
	 * replacement from the last call is used. Setting the replacement to {@code null} removes it.
	 *
	 * <p>The allowed replacements depend on the type of the source. All types accept an
	 * {@link XResForwarder} object, which is usually created with {@link XModuleResources#fwd}.
	 * The resource request will then be forwarded to another {@link android.content.res.Resources}
	 * object. In addition to that, the following replacement types are accepted:
	 *
	 * <table>
	 *     <thead>
	 *     <tr><th>Resource type</th> <th>Additional allowed replacement types (*)</th> <th>Returned from (**)</th></tr>
	 *     </thead>
	 *
	 *     <tbody>
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/animation-resource.html">Animation</a></td>
	 *         <td>&nbsp;<i>none</i></td>
	 *         <td>{@link #getAnimation}</td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/more-resources.html#Bool">Bool</a></td>
	 *         <td>{@link Boolean}</td>
	 *         <td>{@link #getBoolean}</td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/more-resources.html#Color">Color</a></td>
	 *         <td>{@link Integer} (you might want to use {@link Color#parseColor})</td>
	 *         <td>{@link #getColor}<br>
	 *             {@link #getDrawable} (creates a {@link ColorDrawable})<br>
	 *             {@link #getColorStateList} (calls {@link android.content.res.ColorStateList#valueOf})
	 *         </td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/color-list-resource.html">Color State List</a></td>
	 *         <td>{@link android.content.res.ColorStateList}<br>
	 *             {@link Integer} (calls {@link android.content.res.ColorStateList#valueOf})
	 *         </td>
	 *         <td>{@link #getColorStateList}</td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/more-resources.html#Dimension">Dimension</a></td>
	 *         <td>{@link DimensionReplacement} <i>(since v50)</i></td>
	 *         <td>{@link #getDimension}<br>
	 *             {@link #getDimensionPixelOffset}<br>
	 *             {@link #getDimensionPixelSize}
	 *         </td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/drawable-resource.html">Drawable</a>
	 *             (including <a href="http://developer.android.com/tools/projects/index.html#mipmap">mipmap</a>)</td>
	 *         <td>{@link DrawableLoader}<br>
	 *             {@link Integer} (creates a {@link ColorDrawable})
	 *         </td>
	 *         <td>{@link #getDrawable}<br>
	 *             {@link #getDrawableForDensity}
	 *         </td>
	 *     </tr>
	 *
	 *     <tr><td>Fraction</td>
	 *         <td>&nbsp;<i>none</i></td>
	 *         <td>{@link #getFraction}</td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/more-resources.html#Integer">Integer</a></td>
	 *         <td>{@link Integer}</td>
	 *         <td>{@link #getInteger}</td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/more-resources.html#IntegerArray">Integer Array</a></td>
	 *         <td>{@code int[]}</td>
	 *         <td>{@link #getIntArray}</td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/layout-resource.html">Layout</a></td>
	 *         <td>&nbsp;<i>none, but see {@link #hookLayout}</i></td>
	 *         <td>{@link #getLayout}</td>
	 *     </tr>
	 *
	 *     <tr><td>Movie</td>
	 *         <td>&nbsp;<i>none</i></td>
	 *         <td>{@link #getMovie}</td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/string-resource.html#Plurals">Quantity Strings (Plurals)</a></td>
	 *         <td>&nbsp;<i>none</i></td>
	 *         <td>{@link #getQuantityString}<br>
	 *             {@link #getQuantityText}
	 *         </td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/string-resource.html#String">String</a></td>
	 *         <td>{@link String}<br>
	 *             {@link CharSequence} (for styled texts, see also {@link Html#fromHtml})
	 *         </td>
	 *         <td>{@link #getString}<br>
	 *             {@link #getText}
	 *         </td>
	 *     </tr>
	 *
	 *     <tr><td><a href="http://developer.android.com/guide/topics/resources/string-resource.html#StringArray">String Array</a></td>
	 *         <td>{@code String[]}<br>
	 *             {@code CharSequence[]} (for styled texts, see also {@link Html#fromHtml})
	 *         </td>
	 *         <td>{@link #getStringArray}<br>
	 *             {@link #getTextArray}
	 *         </td>
	 *     </tr>
	 *
	 *     <tr><td>XML</td>
	 *         <td>&nbsp;<i>none</i></td>
	 *         <td>{@link #getXml}<br>
	 *             {@link #getQuantityText}
	 *         </td>
	 *     </tr>
	 *
	 *     </tbody>
	 * </table>
	 *
	 * <p>Other resource types, such as
	 * <a href="http://developer.android.com/guide/topics/resources/style-resource.html">styles/themes</a>,
	 * {@linkplain #openRawResource raw resources} and
	 * <a href="http://developer.android.com/guide/topics/resources/more-resources.html#TypedArray">typed arrays</a>
	 * can't be replaced.
	 *
	 * <p><i>
	 *    * Auto-boxing allows you to use literals like {@code 123} where an {@link Integer} is
	 *      accepted, so you don't neeed to call methods like {@link Integer#valueOf(int)} manually.<br>
	 *    ** Some of these methods have multiple variants, only one of them is mentioned here.
	 * </i>
	 *
	 * @param pkg The package name, e.g. {@code com.example.myapplication}.
	 *            See {@link #getResourcePackageName}.
	 * @param type The type name, e.g. {@code string}.
	 *            See {@link #getResourceTypeName}.
	 * @param name The entry name, e.g. {@code app_name}.
	 *            See {@link #getResourceEntryName}.
	 * @param replacement The replacement.
	 */
	public void setReplacement(String pkg, String type, String name, Object replacement) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Sets a replacement for an individual Android framework resource (in the {@code android} package).
	 * See {@link #setSystemWideReplacement(String, String, String, Object)}.
	 *
	 * @param id The ID of the resource which should be replaced.
	 * @param replacement The replacement.
	 */
	public static void setSystemWideReplacement(int id, Object replacement) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Sets a replacement for an individual Android framework resource (in the {@code android} package).
	 * See {@link #setSystemWideReplacement(String, String, String, Object)}.
	 *
	 * @deprecated Use {@link #setSystemWideReplacement(String, String, String, Object)} instead.
	 *
	 * @param fullName The full resource name, e.g. {@code android:string/yes}.
	 *                 See {@link #getResourceName}.
	 * @param replacement The replacement.
	 */
	@Deprecated
	public static void setSystemWideReplacement(String fullName, Object replacement) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Sets a replacement for an individual Android framework resource (in the {@code android} package).
	 *
	 * <p>Some resources are part of the Android framework and can be used in any app. They're
	 * accessible via {@link android.R android.R} and are not bound to a specific
	 * {@link android.content.res.Resources} instance. Such resources can be replaced in
	 * {@link IXposedHookZygoteInit#initZygote initZygote()} for all apps. As there is no
	 * {@link XResources} object easily available in that scope, this static method can be used
	 * to set resource replacements. All other details (e.g. how certain types can be replaced) are
	 * mentioned in {@link #setReplacement(String, String, String, Object)}.
	 *
	 * @param pkg The package name, should always be {@code android} here.
	 *            See {@link #getResourcePackageName}.
	 * @param type The type name, e.g. {@code string}.
	 *            See {@link #getResourceTypeName}.
	 * @param name The entry name, e.g. {@code yes}.
	 *            See {@link #getResourceEntryName}.
	 * @param replacement The replacement.
	 */
	public static void setSystemWideReplacement(String pkg, String type, String name, Object replacement) {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public XmlResourceParser getAnimation(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public boolean getBoolean(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public int getColor(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public ColorStateList getColorStateList(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public float getDimension(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public int getDimensionPixelOffset(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public int getDimensionPixelSize(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public Drawable getDrawable(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public Drawable getDrawable(int id, Theme theme) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public Drawable getDrawableForDensity(int id, int density) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public Drawable getDrawableForDensity(int id, int density, Theme theme) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public float getFraction(int id, int base, int pbase) {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public int getInteger(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public int[] getIntArray(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public XmlResourceParser getLayout(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public Movie getMovie(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public CharSequence getQuantityText(int id, int quantity) throws NotFoundException {
		throw new UnsupportedOperationException();
	}
	// these are handled by getQuantityText:
	// public String getQuantityString(int id, int quantity);
	// public String getQuantityString(int id, int quantity, Object... formatArgs);

	/** @hide */
	@Override
	public String[] getStringArray(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public CharSequence getText(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}
	// these are handled by getText:
	// public String getString(int id);
	// public String getString(int id, Object... formatArgs);

	/** @hide */
	@Override
	public CharSequence getText(int id, CharSequence def) {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public CharSequence[] getTextArray(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/** @hide */
	@Override
	public XmlResourceParser getXml(int id) throws NotFoundException {
		throw new UnsupportedOperationException();
	}

	/**
	 * Generates a fake resource ID.
	 *
	 * <p>The parameter is just hashed, it doesn't have a deeper meaning. However, it's recommended
	 * to use values with a low risk for conflicts, such as a full resource name. Calling this
	 * method multiple times will return the same ID.
	 *
	 * @param resName A used for hashing, see above.
	 * @return The fake resource ID.
	 */
	public static int getFakeResId(String resName) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Generates a fake resource ID.
	 *
	 * <p>This variant uses the result of {@link #getResourceName} to create the hash that the ID is
	 * based on. The given resource doesn't need to match the {@link XResources} instance for which
	 * the fake resource ID is going to be used.
	 *
	 * @param res The {@link android.content.res.Resources} object to be used for hashing.
	 * @param id The resource ID to be used for hashing.
	 * @return The fake resource ID.
	 */
	public static int getFakeResId(Resources res, int id) {
		throw new UnsupportedOperationException();
	}

	/**
	 * Makes any individual resource available from another {@link android.content.res.Resources}
	 * instance available in this {@link XResources} instance.
	 *
	 * <p>This method combines calls to {@link #getFakeResId(Resources, int)} and
	 * {@link #setReplacement(int, Object)} to generate a fake resource ID and set up a replacement
	 * for it which forwards to the given resource.
	 *
	 * <p>The returned ID can only be used to retrieve the resource, it won't work for methods like
	 * {@link #getResourceName} etc.
	 *
	 * @param res The target {@link android.content.res.Resources} instance.
	 * @param id The target resource ID.
	 * @return The fake resource ID (see above).
	 */
	public int addResource(Resources res, int id) {
		throw new UnsupportedOperationException();
	}


	// =======================================================
	//   DrawableLoader class
	// =======================================================
	/**
	 * Callback for drawable replacements. Instances of this class can passed to
	 * {@link #setReplacement(String, String, String, Object)} and its variants.
	 *
	 * <p class="caution">Make sure to always return new {@link Drawable} instances, as drawables
	 * usually can't be reused.
	 */
	@SuppressWarnings("UnusedParameters")
	public static abstract class DrawableLoader {
		/**
		 * Constructor.
		 */
		public DrawableLoader() {}

		/**
		 * Called when the hooked drawable resource has been requested.
		 *
		 * @param res The {@link XResources} object in which the hooked drawable resides.
		 * @param id The resource ID which has been requested.
		 * @return The {@link Drawable} which should be used as replacement. {@code null} is ignored.
		 * @throws Throwable Everything the callback throws is caught and logged.
		 */
		public abstract Drawable newDrawable(XResources res, int id) throws Throwable;

		/**
		 * Like {@link #newDrawable}, but called for {@link #getDrawableForDensity}. The default
		 * implementation is to use the result of {@link #newDrawable}.
		 *
		 * @param res The {@link XResources} object in which the hooked drawable resides.
		 * @param id The resource ID which has been requested.
		 * @param density The desired screen density indicated by the resource as found in
		 *                {@link DisplayMetrics}.
		 * @return The {@link Drawable} which should be used as replacement. {@code null} is ignored.
		 * @throws Throwable Everything the callback throws is caught and logged.
		 */
		public Drawable newDrawableForDensity(XResources res, int id, int density) throws Throwable {
			return newDrawable(res, id);
		}
	}


	// =======================================================
	//   DimensionReplacement class
	// =======================================================
	/**
	 * Callback for dimension replacements. Instances of this class can passed to
	 * {@link #setReplacement(String, String, String, Object)} and its variants.
	 */
	public static class DimensionReplacement {
		private final float mValue;
		private final int mUnit;

		/**
		 * Creates an instance that can be used for {@link #setReplacement(String, String, String, Object)}
		 * to replace a dimension resource.
		 *
		 * @param value The value of the replacement, in the unit specified with the next parameter.
		 * @param unit One of the {@code COMPLEX_UNIT_*} constants in {@link TypedValue}.
		 */
		public DimensionReplacement(float value, int unit) {
			mValue = value;
			mUnit = unit;
		}

		/** Called by {@link android.content.res.Resources#getDimension}. */
		public float getDimension(DisplayMetrics metrics) {
			return TypedValue.applyDimension(mUnit, mValue, metrics);
		}

		/** Called by {@link android.content.res.Resources#getDimensionPixelOffset}. */
		public int getDimensionPixelOffset(DisplayMetrics metrics) {
			return (int) TypedValue.applyDimension(mUnit, mValue, metrics);
		}

		/** Called by {@link android.content.res.Resources#getDimensionPixelSize}. */
		public int getDimensionPixelSize(DisplayMetrics metrics) {
			final float f = TypedValue.applyDimension(mUnit, mValue, metrics);
			final int res = (int)(f+0.5f);
			if (res != 0) return res;
			if (mValue == 0) return 0;
			if (mValue > 0) return 1;
			return -1;
		}
	}
}