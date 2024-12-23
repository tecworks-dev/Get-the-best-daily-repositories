package com.lyihub.privacy_radar.app

import android.app.Activity
import android.app.Application
import android.graphics.Bitmap
import android.os.Bundle
import android.text.TextUtils
import androidx.appcompat.app.AppCompatDelegate
import com.lyihub.privacy_radar.util.TestResultUtil
import java.lang.ref.WeakReference


class App : Application() {
    val TAG = "App"

    companion object {
        private lateinit var instance: App
        fun get() = instance
    }

    var scanImageList = ArrayList<Bitmap?>()

    var hasAlbumPermission = false
    var hasContactsPermissions = false
    var hasCameraPermission = false
    var hasCallLogPermission = false
    var hasSmsPermission = false

    override fun onCreate() {
        super.onCreate()
        instance = this
        //关闭黑夜模式
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
        //充值测试数据
        TestResultUtil.resetTestData()
    }

}