package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.View.OnClickListener
import android.widget.AdapterView
import android.widget.AdapterView.OnItemClickListener
import com.lyihub.privacy_radar.adapter.ContactAdapter
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.util.ContactUtil
import com.lyihub.privacy_radar.util.ResUtils
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import com.lyihub.privacy_radar.util.SpannableUtil
import kotlinx.android.synthetic.main.activity_contact_result.*

class ContactResultActivity : BaseActivity(),OnItemClickListener,OnClickListener {

    companion object {
        fun intentStart (activity: Activity) {
            var intent = Intent(activity, ContactResultActivity::class.java)
            activity.startActivity(intent)
        }
    }

    var mContactAdapter: ContactAdapter? = null

    override fun getLayoutResource() = R.layout.activity_contact_result

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    fun initView() {
        mContactAdapter = ContactAdapter(this,this)
        mRvContact.adapter = mContactAdapter

        mIvBack.setOnClickListener(this)
    }

    fun initData() {
        ContactUtil.getAllContacts(this){
            val count = it.size
            SharedPreferencesUtils.contactCount = count

            var dotTextSize = ResUtils.getDimenPixRes(com.victor.screen.match.library.R.dimen.dp_28)
            val text = "获取联系人\n${count}个"
            val spanText = "个"
            mTvTip.text = SpannableUtil.getSpannableTextSize(dotTextSize,text, spanText)

            mContactAdapter?.showData(it)
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