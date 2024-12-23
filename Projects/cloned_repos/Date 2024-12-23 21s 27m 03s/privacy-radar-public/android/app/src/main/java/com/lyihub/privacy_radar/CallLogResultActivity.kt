package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.View.OnClickListener
import android.widget.AdapterView
import android.widget.AdapterView.OnItemClickListener
import com.lyihub.privacy_radar.adapter.CallLogAdapter
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.util.CallLogUtil
import com.lyihub.privacy_radar.util.ContactUtil
import com.lyihub.privacy_radar.util.ResUtils
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import com.lyihub.privacy_radar.util.SpannableUtil
import kotlinx.android.synthetic.main.activity_call_log_result.*

class CallLogResultActivity : BaseActivity(),OnItemClickListener,OnClickListener {

    companion object {
        fun intentStart (activity: Activity) {
            var intent = Intent(activity, CallLogResultActivity::class.java)
            activity.startActivity(intent)
        }
    }

    var mCallLogAdapter: CallLogAdapter? = null

    override fun getLayoutResource() = R.layout.activity_call_log_result

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    fun initView() {
        mCallLogAdapter = CallLogAdapter(this,this)
        mRvCallLog.adapter = mCallLogAdapter

        mIvBack.setOnClickListener(this)
    }

    fun initData() {
        CallLogUtil.getAllCallLog(this){
            val count = it.size
            SharedPreferencesUtils.callLogCount = count

            var dotTextSize = ResUtils.getDimenPixRes(com.victor.screen.match.library.R.dimen.dp_28)
            val text = "获取通话记录\n${count}条"
            val spanText = "条"
            mTvTip.text = SpannableUtil.getSpannableTextSize(dotTextSize,text, spanText)

            mCallLogAdapter?.showData(it)
        }
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