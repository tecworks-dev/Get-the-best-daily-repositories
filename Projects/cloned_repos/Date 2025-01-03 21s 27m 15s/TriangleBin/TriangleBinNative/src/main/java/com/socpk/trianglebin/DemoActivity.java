package com.socpk.trianglebin;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;

import org.libsdl.app.SDLActivity;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class DemoActivity extends SDLActivity {

    /* A fancy way of getting the class name */
    private static final String TAG = DemoActivity.class.getSimpleName();

    @Override
    protected String[] getLibraries() {
        return new String[]{"hidapi", "SDL2", "demo"};
    }

    @Override
    protected String[] getArguments() {
        return new String[]{getFilesDir().getAbsolutePath()};
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }
}
