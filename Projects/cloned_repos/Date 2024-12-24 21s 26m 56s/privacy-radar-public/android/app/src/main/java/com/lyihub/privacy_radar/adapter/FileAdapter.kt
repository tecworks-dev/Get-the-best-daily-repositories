package com.lyihub.privacy_radar.adapter

import android.content.Context
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.data.CallLogInfo
import com.lyihub.privacy_radar.holder.CallLogContentHolder
import com.lyihub.privacy_radar.holder.FileContentHolder
import java.io.File


class FileAdapter(context: Context, listener: AdapterView.OnItemClickListener) :
    BaseRecycleAdapter<File, RecyclerView.ViewHolder>(context, listener) {

    override fun onCreateHeadVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder? {
        return null
    }

    override fun onBindHeadVHolder(viewHolder: RecyclerView.ViewHolder, data: File?, position: Int) {
    }

    override fun onCreateContentVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return FileContentHolder(inflate(R.layout.rv_file_cell, parent))
    }

    override fun onBindContentVHolder(viewHolder: RecyclerView.ViewHolder, data: File?, position: Int) {
        val contentViewHolder = viewHolder as FileContentHolder
        contentViewHolder.mOnItemClickListener = listener
        contentViewHolder.bindData(data)
    }
}