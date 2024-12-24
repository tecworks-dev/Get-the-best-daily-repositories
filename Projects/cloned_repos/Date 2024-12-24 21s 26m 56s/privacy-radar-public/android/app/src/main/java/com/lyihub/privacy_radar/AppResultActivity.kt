package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.View.OnClickListener
import android.widget.AdapterView
import android.widget.AdapterView.OnItemClickListener
import com.lyihub.privacy_radar.adapter.AppAdapter
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.util.InstallAppUtil
import com.lyihub.privacy_radar.util.ResUtils
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import com.lyihub.privacy_radar.util.SpannableUtil
import kotlinx.android.synthetic.main.activity_app_result.*

class AppResultActivity : BaseActivity(),OnItemClickListener,OnClickListener {

    companion object {
        fun intentStart (activity: Activity) {
            var intent = Intent(activity, AppResultActivity::class.java)
            activity.startActivity(intent)
        }
    }

    var mAppAdapter: AppAdapter? = null

    override fun getLayoutResource() = R.layout.activity_app_result

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    fun initView() {
        mAppAdapter = AppAdapter(this,this)
        mRvContact.adapter = mAppAdapter

        mIvBack.setOnClickListener(this)
    }

    fun initData() {
        InstallAppUtil.getAllInstallApps(this){
            val count = it.size
            SharedPreferencesUtils.installAppCount = count

            var dotTextSize = ResUtils.getDimenPixRes(com.victor.screen.match.library.R.dimen.dp_28)
            val text = "获取已安装app\n${count}个"
            val spanText = "个"
            mTvTip.text = SpannableUtil.getSpannableTextSize(dotTextSize,text, spanText)

            mAppAdapter?.showData(it)
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