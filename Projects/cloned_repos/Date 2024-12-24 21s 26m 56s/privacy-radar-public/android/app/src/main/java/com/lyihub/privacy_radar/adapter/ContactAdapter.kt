package com.lyihub.privacy_radar.adapter

import android.content.Context
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.data.ContactInfo
import com.lyihub.privacy_radar.holder.ContactContentHolder


class ContactAdapter(context: Context, listener: AdapterView.OnItemClickListener) :
    BaseRecycleAdapter<ContactInfo, RecyclerView.ViewHolder>(context, listener) {

    override fun onCreateHeadVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder? {
        return null
    }

    override fun onBindHeadVHolder(viewHolder: RecyclerView.ViewHolder, data: ContactInfo?, position: Int) {
    }

    override fun onCreateContentVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return ContactContentHolder(inflate(R.layout.rv_contact_cell, parent))
    }

    override fun onBindContentVHolder(viewHolder: RecyclerView.ViewHolder, data: ContactInfo?, position: Int) {
        val contentViewHolder = viewHolder as ContactContentHolder
        contentViewHolder.mOnItemClickListener = listener
        contentViewHolder.bindData(data)
    }
}