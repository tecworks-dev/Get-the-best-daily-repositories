package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.View.OnClickListener
import android.widget.AdapterView
import android.widget.AdapterView.OnItemClickListener
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.util.DeviceUtils
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import kotlinx.android.synthetic.main.activity_device_result.*

class DeviceResultActivity : BaseActivity(),OnItemClickListener,OnClickListener {

    companion object {
        fun intentStart (activity: Activity) {
            var intent = Intent(activity, DeviceResultActivity::class.java)
            activity.startActivity(intent)
        }
    }


    override fun getLayoutResource() = R.layout.activity_device_result

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    fun initView() {
        mIvBack.setOnClickListener(this)
    }

    fun initData() {
        var sb = StringBuffer()
        sb.append("IMEI:${DeviceUtils.getUDID(this)}\n")
        sb.append("S/N:${DeviceUtils.getSerialNumber()}\n")
        sb.append("OsVersion:Android ${DeviceUtils.getSysVersion()}\n")
        sb.append("BuildVersion: ${DeviceUtils.getBuildNumber()}\n")
        sb.append("Model: ${DeviceUtils.getPhoneModel()}")
        mTvDeviceInfo.text = sb.toString()

        SharedPreferencesUtils.deviceResult = 1
    }

    override fun onItemClick(p0: AdapterView<*>?, v: View?, position: Int, p3: Long) {
    }

    override fun onClick(v: View?) {
        when(v?.id) {
            R.id.mIvBack -> {
                finish()
            }
        }
    }

}