package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.View.OnClickListener
import android.widget.AdapterView
import android.widget.AdapterView.OnItemClickListener
import com.lyihub.privacy_radar.adapter.FileAdapter
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.util.FileUtil
import com.lyihub.privacy_radar.util.ResUtils
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import com.lyihub.privacy_radar.util.SpannableUtil
import kotlinx.android.synthetic.main.activity_file_result.*

class FileResultActivity : BaseActivity(),OnItemClickListener,OnClickListener {

    companion object {
        fun intentStart (activity: Activity) {
            var intent = Intent(activity, FileResultActivity::class.java)
            activity.startActivity(intent)
        }
    }

    var mFileAdapter: FileAdapter? = null

    override fun getLayoutResource() = R.layout.activity_file_result

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    fun initView() {
        mFileAdapter = FileAdapter(this,this)
        mRvFile.adapter = mFileAdapter

        mIvBack.setOnClickListener(this)
    }

    fun initData() {
        FileUtil.getSdcardFiles{
            val count = it.size
            SharedPreferencesUtils.fileCount = count

            var dotTextSize = ResUtils.getDimenPixRes(com.victor.screen.match.library.R.dimen.dp_28)
            val text = "获取文件\n${count}个"
            val spanText = "个"
            mTvTip.text = SpannableUtil.getSpannableTextSize(dotTextSize,text, spanText)

            mFileAdapter?.showData(it)
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