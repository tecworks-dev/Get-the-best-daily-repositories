package com.lyihub.privacy_radar.holder

import android.content.pm.ApplicationInfo
import android.view.View
import com.lyihub.privacy_radar.data.AppInfo
import com.lyihub.privacy_radar.util.ImageUtils
import kotlinx.android.synthetic.main.rv_app_cell.view.*


class AppContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: AppInfo?) {
        itemView.mIvIcon.setImageDrawable(data?.icon)
        itemView.mTvName.text = data?.name ?: ""
        itemView.mTvPackageName.text = data?.packageName ?: ""
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}