package io.github.duzhaokun123.yapatch;

import android.app.ActivityThread;
import android.app.Application;
import android.app.LoadedApk;
import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.CompatibilityInfo;
import android.os.Build;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.Map;
import java.util.function.BiConsumer;

import de.robv.android.xposed.XposedHelpers;
import io.github.duzhaokun123.yapatch.hooks.SigBypass;
import io.github.duzhaokun123.yapatch.utils.Utils;
import top.canyie.pine.Pine;
import top.canyie.pine.PineConfig;
import top.canyie.pine.xposed.PineXposed;

public class PatchMain {
    static String TAG = "YAPatch";

    private static LoadedApk stubLoadedApk;
    private static LoadedApk appLoadedApk;
    private static ActivityThread activityThread;

    private static PackageManager pm;

    public static void load() throws PackageManager.NameNotFoundException, JSONException, NoSuchMethodException {
        PineConfig.debug = false;
        PineConfig.debuggable = false;

        Pine.ensureInitialized();
        Log.d(TAG, "pine should be initialized");

        activityThread = ActivityThread.currentActivityThread();
        var context = createLoadedApkWithContext();
        if (context == null) {
            Log.e(TAG, "Error when creating context");
            return;
        }
        pm = context.getPackageManager();
        var applicationInfo = pm.getApplicationInfo(context.getPackageName(), PackageManager.GET_META_DATA);
        var config = new JSONObject(applicationInfo.metaData.getString("yapatch"));

        SigBypass.doSigBypass(context, config.getInt("sigbypassLevel"));
        var modules = Utils.fromJsonArray(config.getJSONArray("modules"));

        if (modules.length == 0) {
            Log.w(TAG, "No module to load");
            return;
        }
        for (String module : modules) {
            loadModule(module);
        }
        PineXposed.onPackageLoad(context.getPackageName(), Application.getProcessName(), applicationInfo, true, context.getClassLoader());

        String originalAppComponentFactory = null;
        try {
            originalAppComponentFactory = config.getString("originalAppComponentFactory");
        } catch (JSONException ignored) {
        }
        Log.d(TAG, "originalAppComponentFactory: " + originalAppComponentFactory);
        if (originalAppComponentFactory != null) {
            try {
                context.getClassLoader().loadClass(originalAppComponentFactory);
            } catch (ClassNotFoundException e) { // This will happen on some strange shells like 360
                Log.w(TAG, "Original AppComponentFactory not found: " + originalAppComponentFactory);
            }
        }
    }

    private static Context createLoadedApkWithContext() {
        try {
            var mBoundApplication = XposedHelpers.getObjectField(activityThread, "mBoundApplication");

            stubLoadedApk = (LoadedApk) XposedHelpers.getObjectField(mBoundApplication, "info");
            var appInfo = (ApplicationInfo) XposedHelpers.getObjectField(mBoundApplication, "appInfo");
            var compatInfo = (CompatibilityInfo) XposedHelpers.getObjectField(mBoundApplication, "compatInfo");

            var mPackages = (Map<?, ?>) XposedHelpers.getObjectField(activityThread, "mPackages");
            mPackages.remove(appInfo.packageName);
            appLoadedApk = activityThread.getPackageInfoNoCheck(appInfo, compatInfo);
            XposedHelpers.setObjectField(mBoundApplication, "info", appLoadedApk);

            var activityClientRecordClass = XposedHelpers.findClass("android.app.ActivityThread$ActivityClientRecord", ActivityThread.class.getClassLoader());
            var fixActivityClientRecord = (BiConsumer<Object, Object>) (k, v) -> {
                if (activityClientRecordClass.isInstance(v)) {
                    var pkgInfo = XposedHelpers.getObjectField(v, "packageInfo");
                    if (pkgInfo == stubLoadedApk) {
                        Log.d(TAG, "fix loadedapk from ActivityClientRecord");
                        XposedHelpers.setObjectField(v, "packageInfo", appLoadedApk);
                    }
                }
            };
            var mActivities = (Map<?, ?>) XposedHelpers.getObjectField(activityThread, "mActivities");
            mActivities.forEach(fixActivityClientRecord);
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    var mLaunchingActivities = (Map<?, ?>) XposedHelpers.getObjectField(activityThread, "mLaunchingActivities");
                    mLaunchingActivities.forEach(fixActivityClientRecord);
                }
            } catch (Throwable ignored) {
            }
            Log.i(TAG, "hooked app initialized: " + appLoadedApk);

            var context = (Context) XposedHelpers.callStaticMethod(Class.forName("android.app.ContextImpl"), "createAppContext", activityThread, stubLoadedApk);
            return context;
        } catch (Throwable e) {
            Log.e(TAG, "createLoadedApk", e);
            return null;
        }
    }

    private static void loadModule(String module) {
        ApplicationInfo moduleInfo;
        try {
            moduleInfo = pm.getApplicationInfo(module, 0);
        } catch (PackageManager.NameNotFoundException e) {
            Log.e(TAG, "Module not found: " + module);
            return;
        }

        var modulePath = moduleInfo.sourceDir;
        var librarySearchPath = modulePath + "!/lib/arm64-v8a";
        PineXposed.loadModule(new File(modulePath), librarySearchPath, false);
        Log.d(TAG, "Module loaded: " + module);
    }
}
