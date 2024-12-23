package com.lyihub.privacy_radar.util

import android.content.SharedPreferences
import android.preference.PreferenceManager
import android.util.Log
import com.lyihub.privacy_radar.util.SharedPreferencesUtils.TAG
import com.lyihub.privacy_radar.app.App
import kotlin.properties.ReadWriteProperty
import kotlin.reflect.KProperty


object SharedPreferencesUtils {
    val TAG = "SharedPreferencesUtils"
    /**
     * 创建 SharedPreferences 对象
     */
    var preferences: SharedPreferences = PreferenceManager.getDefaultSharedPreferences(App.get())

    /**
     * 获取相册图片数量 -1 未完成测试
     */
    var albumCount by SharedPreferenceDelegates.int(-1)

    /**
     * 获取联系人数量 -1 未完成测试
     */
    var contactCount by SharedPreferenceDelegates.int(-1)

    /**
     * 获取文件数量 -1 未完成测试
     */
    var fileCount by SharedPreferenceDelegates.int(-1)

    /**
     * 二维码扫码结果 -1 未完成测试 -2 扫码失败或未识别到二维码 1 扫码成功
     */
    var scanResult by SharedPreferenceDelegates.int(-1)

    /**
     * 手机信息获取结果 -1 未完成测试 1 获取成功
     */
    var deviceResult by SharedPreferenceDelegates.int(-1)

    /**
     * 获取通话记录数量 -1 未完成测试
     */
    var callLogCount by SharedPreferenceDelegates.int(-1)

    /**
     * 获取短信记录数量 -1 未完成测试
     */
    var smsCount by SharedPreferenceDelegates.int(-1)

    /**
     * 获取已安装app数量 -1 未完成测试
     */
    var installAppCount by SharedPreferenceDelegates.int(-1)


    fun putInt(key: String, value: Int) {
        Log.e(TAG, "putInt()-$key=$value")
        val ed = preferences.edit()
        ed.putInt(key, value)
        ed.commit()
    }

    fun getInt(key: String): Int {
        val value = preferences.getInt(key, 0)
        Log.e(TAG, "putInt()-$key=$value")
        return value
    }

    fun getInt(key: String, defauleValue: Int): Int {
        val value = preferences.getInt(key, defauleValue)
        Log.e(TAG, "putInt()-$key=$value")
        return value
    }

    fun putLong(key: String, value: Long) {
        Log.e(TAG, "putLong()-$key=$value")
        val ed = preferences.edit()
        ed.putLong(key, value)
        ed.commit()
    }

    fun getLong(key: String): Long? {
        val value = preferences.getLong(key, 0)
        Log.e(TAG, "getLong()-$key=$value")
        return value
    }

    fun putString(key: String, value: String) {
        Log.e(TAG, "putString()-$key=$value")
        val ed = preferences.edit()
        ed.putString(key, value)
        ed.commit()
    }

    fun getString(key: String, defaultValue: String?): String? {
        val value = preferences.getString(key, defaultValue)
        Log.e(TAG, "getString()-$key=$value")
        return value
    }

    fun putBoolean(key: String, value: Boolean) {
        Log.e(TAG, "putBoolean()-$key=$value")
        val ed = preferences.edit()
        ed.putBoolean(key, value)
        ed.commit()
    }

    fun getBoolean(key: String): Boolean {
        val value = preferences.getBoolean(key, false)
        Log.e(TAG, "getBoolean()-$key=$value")
        return value
    }

    fun getBoolean(key: String, defaultValue: Boolean): Boolean {
        val value = preferences.getBoolean(key, defaultValue)
        Log.e(TAG, "getBoolean()-$key=$value")
        return value
    }
}

/**
 * 定义类型 属性委托类
 */
private object SharedPreferenceDelegates {
    /**
     * 定义委托获取和设置对应类型的方法
     * 委托的原理,大家可以看我前面的文章
     */
    fun int(defaultValue: Int = 0) = object : ReadWriteProperty<SharedPreferencesUtils, Int> {

        override fun getValue(thisRef: SharedPreferencesUtils, property: KProperty<*>): Int {
            /**
             * 当获取值的时候,调用此方法
             * key 值是对应变量的昵称
             */
            return SharedPreferencesUtils.preferences.getInt(property.name, defaultValue)
        }

        override fun setValue(thisRef: SharedPreferencesUtils, property: KProperty<*>, value: Int) {
            /**
             * 当设置值的时候,调用此方法
             * key 值是对应变量的昵称
             */
            Log.e(TAG,"int-property.name = " + property.name)
            Log.e(TAG,"int-value = " + value)
            SharedPreferencesUtils.preferences.edit().putInt(property.name, value).apply()
        }
    }

    fun long(defaultValue: Long = 0L) = object : ReadWriteProperty<SharedPreferencesUtils, Long> {

        override fun getValue(thisRef: SharedPreferencesUtils, property: KProperty<*>): Long {
            return SharedPreferencesUtils.preferences.getLong(property.name, defaultValue)
        }

        override fun setValue(
            thisRef: SharedPreferencesUtils,
            property: KProperty<*>,
            value: Long
        ) {
            SharedPreferencesUtils.preferences.edit().putLong(property.name, value).apply()
        }
    }

    fun boolean(defaultValue: Boolean = false) =
        object : ReadWriteProperty<SharedPreferencesUtils, Boolean> {
            override fun getValue(
                thisRef: SharedPreferencesUtils,
                property: KProperty<*>
            ): Boolean {
                return SharedPreferencesUtils.preferences.getBoolean(property.name, defaultValue)
            }

            override fun setValue(
                thisRef: SharedPreferencesUtils,
                property: KProperty<*>,
                value: Boolean
            ) {
                SharedPreferencesUtils.preferences.edit().putBoolean(property.name, value).apply()
            }
        }

    fun float(defaultValue: Float = 0.0f) =
        object : ReadWriteProperty<SharedPreferencesUtils, Float> {
            override fun getValue(thisRef: SharedPreferencesUtils, property: KProperty<*>): Float {
                return SharedPreferencesUtils.preferences.getFloat(property.name, defaultValue)
            }

            override fun setValue(
                thisRef: SharedPreferencesUtils,
                property: KProperty<*>,
                value: Float
            ) {
                SharedPreferencesUtils.preferences.edit().putFloat(property.name, value).apply()
            }
        }

    fun string(defaultValue: String) = object : ReadWriteProperty<SharedPreferencesUtils, String> {
        override fun getValue(thisRef: SharedPreferencesUtils, property: KProperty<*>): String {
            return SharedPreferencesUtils.preferences.getString(property.name, defaultValue) ?: ""
        }

        override fun setValue(
            thisRef: SharedPreferencesUtils,
            property: KProperty<*>,
            value: String
        ) {
            Log.e(TAG,"string-property.name = " + property.name)
            Log.e(TAG,"string-value = " + value)
            SharedPreferencesUtils.preferences.edit().putString(property.name, value).apply()
        }
    }

}