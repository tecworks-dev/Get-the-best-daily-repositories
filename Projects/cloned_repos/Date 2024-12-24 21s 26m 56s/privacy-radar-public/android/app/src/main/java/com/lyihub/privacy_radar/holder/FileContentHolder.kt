package com.lyihub.privacy_radar.holder

import android.view.View
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.data.CallLogInfo
import com.lyihub.privacy_radar.util.ImageUtils
import kotlinx.android.synthetic.main.rv_file_cell.view.*
import java.io.File


class FileContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: File?) {
        val isDirectory = data?.isDirectory ?: false
        if (isDirectory) {
            ImageUtils.instance.loadImage(itemView.context,itemView.mIvFileType, R.mipmap.ic_zfile_folder)
        } else {
            ImageUtils.instance.loadImage(itemView.context,itemView.mIvFileType, R.mipmap.ic_zfile_other)
        }
        itemView.mTvName.text = data?.name ?: ""
        itemView.mIvFilePath.text = data?.absolutePath ?: ""
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}