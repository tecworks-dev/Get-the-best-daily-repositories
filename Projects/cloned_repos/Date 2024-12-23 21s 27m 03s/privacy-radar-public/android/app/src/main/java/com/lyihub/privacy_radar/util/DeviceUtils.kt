package com.lyihub.privacy_radar.util

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import android.telephony.TelephonyManager
import android.text.TextUtils
import android.util.Log
import androidx.core.app.ActivityCompat
import java.io.BufferedReader
import java.io.FileReader
import java.io.IOException
import java.net.Inet4Address
import java.net.InetAddress
import java.net.NetworkInterface
import java.net.SocketException
import java.util.Locale
import java.util.UUID
import kotlin.experimental.and



object DeviceUtils {
    const val TAG = "DeviceUtils"

    fun getUDID(context: Context?): String {
        var udid: String = getDeviceID(context)

        if (TextUtils.isEmpty(udid)) {
            udid = getDeviceUUID()
        }
        if (TextUtils.isEmpty(udid)) {
            udid = getAndroidID(context)
        }
        var udidMd5Str = CryptoUtils.MD5(udid)
        Log.e(TAG,"getUDID()...udidMd5Str = $udidMd5Str")
        return udidMd5Str ?: ""
    }

    @SuppressLint("MissingPermission")
    fun getDeviceID (context: Context?): String {
        var deviceId = "unknown"
        try {
            val tm = context?.applicationContext?.getSystemService(
                Context.TELEPHONY_SERVICE) as TelephonyManager

            if (PermissionHelper.hasPermission(context,Manifest.permission.READ_PHONE_STATE)) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    deviceId = tm.getImei(0)
                } else {
                    deviceId = tm.deviceId
                }
            } else {
                Log.e(TAG,"getDeviceID()...not has READ_PHONE_STATE permission")
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        Log.e(TAG,"getDeviceID()...deviceId = $deviceId")
        return deviceId
    }

    fun getAndroidID (context: Context?): String {
        var androidId = ""
        try {
            androidId = Settings.Secure.getString(
                context?.contentResolver, Settings.Secure.ANDROID_ID)
        } catch (e: Exception) {
            e.printStackTrace()
        }
        Log.e(TAG,"getAndroidID()...androidId = $androidId")
        return androidId
    }

    fun getDeviceUUID(): String {
        var brand = Build.BRAND

        val dev = Build.ID +
                Build.MANUFACTURER +
                Build.BRAND +
                Build.PRODUCT +
                Build.DEVICE +
                Build.BOARD +
                Build.DISPLAY +
                Build.MODEL +
                Build.FINGERPRINT +
                Build.HOST
        var uuid = UUID(dev.hashCode().toLong(),brand.hashCode().toLong()).toString()
        Log.e(TAG,"getDeviceUUID()...uuid = $uuid")
        return uuid
    }

    fun getMac(): String {
        var macAddr = ""
        try {
            val mac: ByteArray
            val ne = NetworkInterface.getByInetAddress(
                InetAddress.getByName(getLocalIpAddress())
            )
            mac = ne.hardwareAddress
            macAddr = byte2hex(mac)
        } catch (e: Exception) {
            e.printStackTrace()
        }
        Log.e(TAG,"getMac()...macAddr = $macAddr")
        return macAddr
    }

    fun getEthernetMac(): String? {
        var reader: BufferedReader? = null
        var ethernetMac = ""
        try {
            reader = BufferedReader(
                FileReader(
                    "sys/class/net/eth0/address"
                )
            )
            ethernetMac = reader.readLine()
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            try {
                reader?.close()
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        Log.e(TAG,"getEthernetMac()...ethernetMac = $ethernetMac")
        return ethernetMac.toUpperCase()
    }

    fun byte2hex(b: ByteArray): String {
        var hs = StringBuffer(b.size)
        var stmp = ""
        val len = b.size
        for (n in 0 until len) {
            stmp = Integer.toHexString((b[n] and 0xFF.toByte()).toInt())
            hs = if (stmp.length == 1) hs.append("0").append(stmp) else {
                hs.append(stmp)
            }
        }
        return hs.toString()
    }


    fun getLocalIpAddress(): String? {
        var sLocalIPAddress = ""
        try {
            val en =
                NetworkInterface.getNetworkInterfaces()
            while (en.hasMoreElements()) {
                val netInterface =
                    en.nextElement() as NetworkInterface
                val ipaddr =
                    netInterface.inetAddresses
                while (ipaddr.hasMoreElements()) {
                    val inetAddress =
                        ipaddr.nextElement() as InetAddress
                    if (!inetAddress.isLoopbackAddress && inetAddress is Inet4Address) {
                        sLocalIPAddress = inetAddress.getHostAddress().toString()
                    }
                }
            }
        } catch (ex: SocketException) {
            ex.printStackTrace()
        }
        Log.e(TAG,"getLocalIpAddress()...sLocalIPAddress = $sLocalIPAddress")
        return sLocalIPAddress
    }

    /**
     * 获取手机品牌
     *
     * @return
     */
    fun getPhoneBrand(): String? {
        return Build.BRAND
    }

    /**
     * 获取系统版本
     *
     * @return
     */
    fun getSysVersion(): String? {
        return Build.VERSION.RELEASE
    }

    fun getSdkInt(): Int {
        return Build.VERSION.SDK_INT
    }

    /**
     * 获取手机型号
     *
     * @return
     */
    fun getPhoneModel(): String? {
        return Build.MODEL
    }

    fun getBuildNumber(): String? {
        return Build.DISPLAY
    }

    fun getSysLanguage(): String? {
        val locale = Locale.getDefault()
        return locale.language
    }

    fun getSerialNumber(): String {
        var serial = "unknown"
        try {
            serial = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                Build.getSerial()
            } else {
                Build.SERIAL
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return serial
    }
}