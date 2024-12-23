package com.lyihub.privacy_radar.util

import android.util.Log
import com.lyihub.privacy_radar.data.TestReportInfo

object TestResultUtil {

    fun resetTestData() {
        SharedPreferencesUtils.albumCount = -1
        SharedPreferencesUtils.contactCount = -1
        SharedPreferencesUtils.fileCount = -1
        SharedPreferencesUtils.scanResult = -1
        SharedPreferencesUtils.deviceResult = -1
        SharedPreferencesUtils.callLogCount = -1
        SharedPreferencesUtils.smsCount = -1
        SharedPreferencesUtils.installAppCount = -1
    }

    fun isTestCompleted(): Boolean {
        val albumCount = SharedPreferencesUtils.albumCount
        val contactCount = SharedPreferencesUtils.contactCount
        val fileCount = SharedPreferencesUtils.fileCount
        val scanResult = SharedPreferencesUtils.scanResult
        val deviceResult = SharedPreferencesUtils.deviceResult
        val callLogCount = SharedPreferencesUtils.callLogCount
        val smsCount = SharedPreferencesUtils.smsCount
        val installAppCount = SharedPreferencesUtils.installAppCount
        Log.e(javaClass.simpleName,"isTestCompleted-albumCount = $albumCount")
        Log.e(javaClass.simpleName,"isTestCompleted-contactCount = $contactCount")
        Log.e(javaClass.simpleName,"isTestCompleted-fileCount = $fileCount")
        Log.e(javaClass.simpleName,"isTestCompleted-scanResult = $scanResult")
        Log.e(javaClass.simpleName,"isTestCompleted-deviceResult = $deviceResult")
        Log.e(javaClass.simpleName,"isTestCompleted-callLogCount = $callLogCount")
        Log.e(javaClass.simpleName,"isTestCompleted-smsCount = $smsCount")
        Log.e(javaClass.simpleName,"isTestCompleted-installAppCount = $installAppCount")

        return albumCount >= 0 && contactCount >= 0 &&
                fileCount >= 0 && scanResult >= 0 && deviceResult >= 0
                && callLogCount >= 0 && smsCount >= 0 && installAppCount >= 0
    }

    fun getTestReport(): List<TestReportInfo> {
        var reportList = ArrayList<TestReportInfo>()

        val albumCount = SharedPreferencesUtils.albumCount
        val contactCount = SharedPreferencesUtils.contactCount
        val fileCount = SharedPreferencesUtils.fileCount
        val scanResult = SharedPreferencesUtils.scanResult
        val deviceResult = SharedPreferencesUtils.deviceResult
        val callLogCount = SharedPreferencesUtils.callLogCount
        val smsCount = SharedPreferencesUtils.smsCount
        val installAppCount = SharedPreferencesUtils.installAppCount

        var item1 = TestReportInfo()
        item1.resultType = 0
        item1.content = "已获取相册照片"
        item1.result = albumCount
        item1.resultUnit = "张"

        var item2 = TestReportInfo()
        item2.resultType = 0
        item2.content = "已获取联系人"
        item2.result = contactCount
        item2.resultUnit = "个"

        var item3 = TestReportInfo()
        item3.resultType = 0
        item3.content = "已获取文件"
        item3.result = fileCount
        item3.resultUnit = "个"

        var item4 = TestReportInfo()
        item4.resultType = 1
        item4.content = "二维码扫描结果"
        item4.result = scanResult

        var item5 = TestReportInfo()
        item5.resultType = 1
        item5.content = "手机信息"
        item5.result = deviceResult

        var item6 = TestReportInfo()
        item6.resultType = 0
        item6.content = "已获取通话记录"
        item6.result = callLogCount
        item6.resultUnit = "条"

        var item7 = TestReportInfo()
        item7.resultType = 0
        item7.content = "已获取短信"
        item7.result = smsCount
        item7.resultUnit = "条"

        var item8 = TestReportInfo()
        item8.resultType = 0
        item8.content = "获取已安装app"
        item8.result = installAppCount
        item8.resultUnit = "个"

        reportList.add(item1)
        reportList.add(item2)
        reportList.add(item3)
        reportList.add(item4)
        reportList.add(item5)
        reportList.add(item6)
        reportList.add(item7)
        reportList.add(item8)

        return reportList
    }
}