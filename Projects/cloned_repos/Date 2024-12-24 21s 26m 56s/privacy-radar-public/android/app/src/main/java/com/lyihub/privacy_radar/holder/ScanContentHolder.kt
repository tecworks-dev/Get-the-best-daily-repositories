package com.lyihub.privacy_radar.holder

import android.graphics.Bitmap
import android.view.View
import com.lyihub.privacy_radar.util.ImageUtils
import kotlinx.android.synthetic.main.rv_scan_cell.view.*


class ScanContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: Bitmap?) {
        itemView.mIvAlbum.setImageBitmap(data)
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}