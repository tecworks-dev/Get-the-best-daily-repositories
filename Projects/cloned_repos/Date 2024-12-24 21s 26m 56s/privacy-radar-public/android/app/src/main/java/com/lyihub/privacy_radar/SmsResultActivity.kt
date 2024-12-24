package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.View.OnClickListener
import android.widget.AdapterView
import android.widget.AdapterView.OnItemClickListener
import com.lyihub.privacy_radar.adapter.SmsAdapter
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.util.CallLogUtil
import com.lyihub.privacy_radar.util.ResUtils
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import com.lyihub.privacy_radar.util.SmsUtil
import com.lyihub.privacy_radar.util.SpannableUtil
import kotlinx.android.synthetic.main.activity_sms_result.*

class SmsResultActivity : BaseActivity(),OnItemClickListener,OnClickListener {

    companion object {
        fun intentStart (activity: Activity) {
            var intent = Intent(activity, SmsResultActivity::class.java)
            activity.startActivity(intent)
        }
    }

    var mSmsAdapter: SmsAdapter? = null

    override fun getLayoutResource() = R.layout.activity_sms_result

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    fun initView() {
        mSmsAdapter = SmsAdapter(this,this)
        mRvSms.adapter = mSmsAdapter

        mIvBack.setOnClickListener(this)
    }

    fun initData() {
        SmsUtil.getAllSms(this){
            val count = it.size
            SharedPreferencesUtils.smsCount = count

            var dotTextSize = ResUtils.getDimenPixRes(com.victor.screen.match.library.R.dimen.dp_28)
            val text = "获取短信\n${count}条"
            val spanText = "条"
            mTvTip.text = SpannableUtil.getSpannableTextSize(dotTextSize,text, spanText)

            mSmsAdapter?.showData(it)
        }
    }

    override fun onItemClick(p0: AdapterView<*>?, v: View?, position: Int, p3: Long) {
        val data = mSmsAdapter?.getItem(position)
        SmsDetailActivity.intentStart(this,data)
    }

    override fun onClick(v: View?) {
        when(v?.id) {
            R.id.mIvBack -> {
                finish()
            }
        }
    }

}