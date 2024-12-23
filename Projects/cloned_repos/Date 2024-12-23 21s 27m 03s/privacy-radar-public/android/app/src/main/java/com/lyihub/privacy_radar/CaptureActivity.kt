package com.lyihub.privacy_radar

import android.os.Bundle
import android.os.PersistableBundle
import android.util.Log
import android.view.KeyEvent
import android.view.View
import android.view.View.OnClickListener
import com.journeyapps.barcodescanner.CaptureManager
import com.journeyapps.barcodescanner.SourceData
import com.journeyapps.barcodescanner.interfaces.OnZxingDecodeListener
import com.lyihub.privacy_radar.app.App
import com.lyihub.privacy_radar.base.BaseActivity
import kotlinx.android.synthetic.main.activity_capture.*


class CaptureActivity : BaseActivity(),OnClickListener,OnZxingDecodeListener {

    private var capture: CaptureManager? = null
    override fun getLayoutResource() = R.layout.activity_capture

    override fun onCreate(savedInstanceState: Bundle?) {
        statusBarTextColorBlack = false
        super.onCreate(savedInstanceState)
        initView(savedInstanceState)
    }

    fun initView(savedInstanceState: Bundle?) {
        App.get().scanImageList.clear()

        capture = CaptureManager(this, barcodeScannerView)
        capture?.setOnZxingDecodeListener(this)
        capture?.initializeFromIntent(intent, savedInstanceState)
        capture?.decode()

        mIvBack.setOnClickListener(this)
    }

    override fun onResume() {
        super.onResume()
        capture?.onResume()
    }

    override fun onPause() {
        super.onPause()
        capture?.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        capture?.onDestroy()
    }

    override fun onSaveInstanceState(outState: Bundle, outPersistentState: PersistableBundle) {
        super.onSaveInstanceState(outState, outPersistentState)
        capture?.onSaveInstanceState(outState)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        capture?.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return barcodeScannerView.onKeyDown(keyCode, event) || super.onKeyDown(keyCode, event)
    }

    override fun onClick(v: View?) {
        when (v?.id) {
            R.id.mIvBack -> {
                finish()
            }
        }
    }

    override fun OnZxingDecode(sourceData: SourceData?) {
        Log.e(TAG,"OnZxingDecode--bm = ${sourceData?.bitmap}")
        App.get().scanImageList.add(sourceData?.bitmap)
    }
}