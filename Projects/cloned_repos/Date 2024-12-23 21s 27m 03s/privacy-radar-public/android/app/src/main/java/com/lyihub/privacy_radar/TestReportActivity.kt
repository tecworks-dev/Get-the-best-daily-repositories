package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.View.OnClickListener
import com.lyihub.privacy_radar.adapter.TestReportAdapter
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.util.TestResultUtil
import kotlinx.android.synthetic.main.activity_test_report.*

class TestReportActivity : BaseActivity(),OnClickListener {

    companion object {
        fun intentStart (activity: Activity) {
            var intent = Intent(activity, TestReportActivity::class.java)
            activity.startActivity(intent)
        }
    }

    var mTestReportAdapter: TestReportAdapter? = null

    override fun getLayoutResource() = R.layout.activity_test_report

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    fun initView() {
        mTestReportAdapter = TestReportAdapter(this,null)
        mRvReport.adapter = mTestReportAdapter

        mIvBack.setOnClickListener(this)
    }

    fun initData() {
        var reportList = TestResultUtil.getTestReport()
        mTestReportAdapter?.showData(reportList)
    }

    override fun onClick(v: View?) {
        when(v?.id) {
            R.id.mIvBack -> {
                finish()
            }
        }
    }
}