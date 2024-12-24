package com.lyihub.privacy_radar.holder

import android.view.View
import com.lyihub.privacy_radar.util.ImageUtils
import kotlinx.android.synthetic.main.rv_album_cell.view.*


class AlbumContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: String?) {
        ImageUtils.instance.loadImage(itemView.context,itemView.mIvAlbum,data)
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}