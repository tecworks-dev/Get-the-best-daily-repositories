package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.View.OnClickListener
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.data.SmsInfo
import com.lyihub.privacy_radar.util.Constant
import kotlinx.android.synthetic.main.activity_sms_detail.*

class SmsDetailActivity : BaseActivity(),OnClickListener {

    companion object {
        fun intentStart (activity: Activity,data: SmsInfo?) {
            var intent = Intent(activity, SmsDetailActivity::class.java)
            intent.putExtra(Constant.INTENT_DATA_KEY,data)
            activity.startActivity(intent)
        }
    }

    var mSmsInfo: SmsInfo? = null

    override fun getLayoutResource() = R.layout.activity_sms_detail

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData(intent)
    }

    fun initView() {
        mIvBack.setOnClickListener(this)
    }

    fun initData(intent: Intent?) {
        mSmsInfo = intent?.getSerializableExtra(Constant.INTENT_DATA_KEY) as SmsInfo?
        mTvNumber.text = mSmsInfo?.address ?: ""
        mTvContent.text = mSmsInfo?.body ?: ""
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        initData(intent)
    }

    override fun onClick(v: View?) {
        when (v?.id) {
            R.id.mIvBack -> {
                finish()
            }
        }
    }
}