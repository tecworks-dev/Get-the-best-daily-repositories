package com.lyihub.privacy_radar

import android.app.Activity
import android.content.Intent
import android.database.ContentObserver
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Message
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.View.OnClickListener
import android.widget.AdapterView
import android.widget.AdapterView.OnItemClickListener
import com.lyihub.privacy_radar.adapter.AlbumAdapter
import com.lyihub.privacy_radar.base.BaseActivity
import com.lyihub.privacy_radar.dialog.CommonTipDialog
import com.lyihub.privacy_radar.interfaces.OnDialogOkCancelClickListener
import com.lyihub.privacy_radar.util.AlbumUtil
import com.lyihub.privacy_radar.util.Constant
import com.lyihub.privacy_radar.util.MainHandler
import com.lyihub.privacy_radar.util.NotificationUtil
import com.lyihub.privacy_radar.util.PermissionHelper
import com.lyihub.privacy_radar.util.ResUtils
import com.lyihub.privacy_radar.util.SharedPreferencesUtils
import com.lyihub.privacy_radar.util.SpannableUtil
import com.lyihub.privacy_radar.util.ToastUtils
import kotlinx.android.synthetic.main.activity_album_result.*


class AlbumResultActivity : BaseActivity(),OnItemClickListener,OnClickListener {

    companion object {
        fun intentStart (activity: Activity) {
            var intent = Intent(activity, AlbumResultActivity::class.java)
            activity.startActivity(intent)
        }
    }

    private lateinit var uri: Uri
    private lateinit var observer: AlbumContentObserver

    var mAlbumAdapter: AlbumAdapter? = null
    var mCommonTipDialog: CommonTipDialog? = null
    var isVisibleToUser = false

    override fun getLayoutResource() = R.layout.activity_album_result

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initView()
        initData()
    }

    override fun onPause() {
        super.onPause()
        isVisibleToUser = false
    }

    override fun onResume() {
        super.onResume()
        isVisibleToUser = true
        MainHandler.get().removeCallbacks(mRefreshRunnable)
        MainHandler.get().post(mRefreshRunnable)
    }
    fun initView() {
        initObserver()
        mAlbumAdapter = AlbumAdapter(this,this)
        mRvAlbum.adapter = mAlbumAdapter

        mIvBack.setOnClickListener(this)

        val isNotifyOpen = PermissionHelper.isNotificationEnabled(this)
        if (!isNotifyOpen) {
            PermissionHelper.openNotification(this)
        }
    }

    fun initData() {
        AlbumUtil.getAllAlbumImages(this){
            showAlbumData(it)
        }
    }

    fun showAlbumData(datas: List<String>) {
        val count = datas.size
        SharedPreferencesUtils.albumCount = count

        var dotTextSize = ResUtils.getDimenPixRes(com.victor.screen.match.library.R.dimen.dp_28)
        val text = "获取相册\n${count}张"
        val spanText = "张"
        mTvTip.text = SpannableUtil.getSpannableTextSize(dotTextSize,text, spanText)

        mAlbumAdapter?.showData(datas)
    }

    fun initObserver() {
        uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        observer = AlbumContentObserver(Handler())
        // 注册内容观察者
        contentResolver.registerContentObserver(uri, true, observer)
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

    inner class AlbumContentObserver internal constructor(handler: Handler?) :
        ContentObserver(handler) {
        override fun onChange(selfChange: Boolean) {
            // 当相册内容发生变化时，发送最新的数据
            Log.e(TAG,"onChange...8888888")
            MainHandler.get().removeCallbacks(mRefreshRunnable)
            MainHandler.get().postDelayed(mRefreshRunnable,2000)
        }
    }

    override fun onDestroy() {
        contentResolver.unregisterContentObserver(observer)
        MainHandler.get().removeCallbacksAndMessages(0)
        super.onDestroy()
    }

    fun showAlbumUpdateTipDialog() {
        Log.e(TAG,"showAlbumUpdateTipDialog-isVisibleToUser = $isVisibleToUser")
        if (!isVisibleToUser) {
            NotificationUtil.sendNotyfy(this)
            return
        }
        if (mCommonTipDialog == null) {
            mCommonTipDialog = CommonTipDialog(this)
            mCommonTipDialog?.mTitle = "相册更新了确定刷新吗？"
            mCommonTipDialog?.mOkText = "确定"
            mCommonTipDialog?.mCancelText = "取消"
            mCommonTipDialog?.mOnDialogOkCancelClickListener = object :
                OnDialogOkCancelClickListener {
                override fun OnDialogOkClick() {
                    initData()
                }

                override fun OnDialogCancelClick() {
                }
            }
        }
        mCommonTipDialog?.show()
    }

    var mRefreshRunnable = Runnable {
        AlbumUtil.getAllAlbumImages(this){
            val count = it.size
            SharedPreferencesUtils.albumCount = count

            val currentCount = mAlbumAdapter?.getContentItemCount() ?: 0
            if (currentCount < count) {
                showAlbumUpdateTipDialog()
            }
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        initData()
    }

}