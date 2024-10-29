package io.github.duzhaokun123.yapatch;

import android.util.Log;

public class AppComponentFactory extends android.app.AppComponentFactory {
    static String TAG = "YAPatch";
    static {
        Log.d(TAG, "AppComponentFactory loaded");
        try {
            PatchMain.load();
        } catch (Exception e) {
            Log.e(TAG, "Error loading patch", e);
        }
    }
}
