package com.lyihub.privacy_radar

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Message
import android.text.TextUtils
import android.util.Log
import android.view.View
import android.view.View.OnClickListener
import com.google.zxing.integration.android.IntentIntegrator
import com.journeyapps.barcodescanner.SourceData
import com.lyihub.privacy_radar.adapter.ScanAdapter
import com.lyihub.privacy_radar.app.App
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.util.ImageUtils
import com.lyihub.privacy_radar.util.MainHandler
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import kotlinx.android.synthetic.main.activity_camera.*

class CameraActivity : BaseActivity(),OnClickListener {

    lateinit var integrator: IntentIntegrator

    var mScanAdapter: ScanAdapter? = null

    override fun getLayoutResource() = R.layout.activity_camera

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initScan()
        initView()
    }

    fun initView() {
        mScanAdapter = ScanAdapter(this,null)
        mRvScan.adapter = mScanAdapter

        mIvBack.setOnClickListener(this)
        mTvSan.setOnClickListener(this)
    }

    fun initScan() {
        integrator = IntentIntegrator(this)
        integrator.setDesiredBarcodeFormats(IntentIntegrator.QR_CODE) // 设置只扫描QR码
//        integrator.setPrompt("请扫描二维码") // 设置扫描提示
        integrator.setPrompt("   ")// 设置扫描提示
        integrator.setCameraId(0) // 使用设备的特定相机（这里是第一个相机）
        integrator.setBeepEnabled(true)// 禁用扫描成功时的蜂鸣声
        integrator.setOrientationLocked(false)// 锁定扫描器的方向
        integrator.setBarcodeImageEnabled(true)// 禁用条形码图像的显示
        integrator.captureActivity = CaptureActivity::class.java
        integrator.initiateScan() // 发起扫描
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        //获取扫码结果
        val result = IntentIntegrator.parseActivityResult(requestCode, resultCode, data)
        if (TextUtils.isEmpty(result.contents)) {
            finish()
            return
        }
        mTvScanResult.text = result.contents

        Log.e(TAG,"onActivityResult-result = ${result}")
//        ImageUtils.instance.loadImage(this,mIvScanImage,result.barcodeImagePath)

        mScanAdapter?.showData(App.get().scanImageList)

        if (TextUtils.isEmpty(result.contents)) {
            SharedPreferencesUtils.scanResult = -2
        } else {
            SharedPreferencesUtils.scanResult = 1
        }

    }

    override fun onClick(v: View?) {
        when(v?.id) {
            R.id.mIvBack -> {
                finish()
            }
            R.id.mTvSan -> {
                integrator.initiateScan() // 发起扫描
            }
        }
    }
}