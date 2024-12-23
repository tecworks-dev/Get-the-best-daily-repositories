package com.lyihub.privacy_radar.adapter

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import androidx.recyclerview.widget.RecyclerView
import com.lyihub.privacy_radar.R
import com.lyihub.privacy_radar.holder.BottomViewHolder
import com.lyihub.privacy_radar.holder.HeaderViewHolder

abstract class BaseRecycleAdapter<T,VH: RecyclerView.ViewHolder>(
    var context: Context?, var listener:AdapterView.OnItemClickListener?):
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    var TAG = javaClass.simpleName

    companion object {
        val LOADING = 0x0001//正在加载
        val LOADING_COMPLETE = 0x0002//加载完毕
        val LOADING_END = 0x0003
    }
    var mDatas: ArrayList<T> = ArrayList()
    var mHeaderCount = 0//头部View个数
    var mBottomCount = 0//底部View个数
    var ITEM_TYPE_HEADER = 0
    var ITEM_TYPE_CONTENT = 1
    var ITEM_TYPE_BOTTOM = 2
    private var loadState = LOADING_COMPLETE//上拉加载状态

    private var isHeaderVisible = false
    private var isFooterVisible = false

    var dataPositionMap = LinkedHashMap<Int,Int>()//有序的map可以按照list选中顺序取出
    var dataMap = LinkedHashMap<Int,T?>()//有序的map可以按照list选中顺序取出

    abstract fun onCreateHeadVHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder?
    abstract fun onBindHeadVHolder(viewHolder: VH, data: T?, position: Int)

    abstract fun onCreateContentVHolder(parent: ViewGroup, viewType: Int): VH
    abstract fun onBindContentVHolder(viewHolder: VH, data: T?, position: Int)

    fun setHeaderVisible(visible: Boolean) {
        isHeaderVisible = visible
        mHeaderCount = 1
        if (!isHeaderVisible) {
            mHeaderCount = 0
        }
    }

    fun setFooterVisible(visible: Boolean) {
        isFooterVisible = visible
        mBottomCount = 1
        if (!isFooterVisible) {
            mBottomCount = 0
        }
    }

    fun inflate(layoutId: Int,parent: ViewGroup): View {
        var inflater = LayoutInflater.from(context)
        return inflater.inflate(layoutId,parent, false)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        if (viewType == ITEM_TYPE_HEADER) {
            return onCreateHeadVHolder(parent, viewType)!!
        } else if (viewType == ITEM_TYPE_CONTENT) {
            return onCreateContentVHolder(parent, viewType)
        } else if (viewType == ITEM_TYPE_BOTTOM) {
            return BottomViewHolder(inflate(R.layout.recyclerview_foot, parent))
        }
        return onCreateContentVHolder(parent, viewType)
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        var item = getItem(position)
        if (holder is HeaderViewHolder) {
            onBindHeadVHolder(holder as VH, item, position)
        } else if (holder is BottomViewHolder) {
            setFooterViewState(holder)
        } else {
            if (isHeaderVisible) {
                item = getItem(position - 1)
                onBindContentVHolder(holder as VH, item, position - 1)
            } else {
                onBindContentVHolder(holder as VH, item, position)
            }
        }
    }

    override fun getItemViewType(position: Int): Int {
        var ITEM_TYPE = ITEM_TYPE_CONTENT
        val dataItemCount = getContentItemCount()
        if (mHeaderCount != 0 && position < mHeaderCount) {//头部View
            ITEM_TYPE = ITEM_TYPE_HEADER
        } else if (mBottomCount != 0 && position >= mHeaderCount + dataItemCount) {//底部View
            ITEM_TYPE = ITEM_TYPE_BOTTOM
        }
        return ITEM_TYPE
    }

    fun isHeaderView(position: Int): Boolean {
        return mHeaderCount != 0 && position < mHeaderCount
    }

    fun isBottomView(position: Int): Boolean {
        return mBottomCount != 0 && position >= mHeaderCount + getContentItemCount()
    }

    fun getContentItemCount(): Int {
        return if (mDatas == null) 0 else mDatas.size
    }

    override fun getItemCount(): Int {
        return mHeaderCount + getContentItemCount() + mBottomCount
    }

    private fun setFooterViewState(bottomViewHolder: BottomViewHolder) {
        when (loadState) {
            LOADING -> {
                bottomViewHolder.progressBar?.visibility = View.VISIBLE
                bottomViewHolder.mTvTitle?.visibility = View.VISIBLE
                bottomViewHolder.mLayoutEnd?.visibility = View.GONE
            }
            LOADING_COMPLETE -> {
                bottomViewHolder.progressBar?.visibility = View.GONE
                bottomViewHolder.mTvTitle?.visibility = View.GONE
                bottomViewHolder.mLayoutEnd?.visibility = View.GONE
            }
            LOADING_END -> {
                bottomViewHolder.progressBar?.visibility = View.GONE
                bottomViewHolder.mTvTitle?.visibility = View.GONE
                bottomViewHolder.mLayoutEnd?.visibility = View.VISIBLE
            }
        }
    }

    /**
     * 设置上拉加载状态
     *
     * @param loadState 0.正在加载 1.加载完成 2.加载到底
     */
    fun setLoadState(loadState: Int) {
        this.loadState = loadState
    }

    /**
     * 获取元素
     *
     * @param position
     * @return
     */
    fun getItem(position: Int): T? {
        //防止越界
        val index = if (position >= 0 && position < mDatas.size) position else 0
        return if (mDatas == null || mDatas.size == 0) {
            null
        } else mDatas.get(index)
    }

    /**
     * 添加元素
     *
     * @param item
     */
    fun add(item: T?) {
        if (item != null) {
            mDatas.add(item)
        }
    }

    /**
     * 添加元素
     *
     * @param item
     */
    fun add(index: Int, item: T?) {
        if (item != null) {
            mDatas.add(index, item)
        }
    }

    fun add(items: List<T>?) {
        if (items != null) {
            mDatas.addAll(items)
        }
    }

    /**
     * 重置元素
     *
     * @param items
     */
    fun setDatas(items: List<T>) {
        mDatas.clear()
        add(items)
    }

    /**
     * 移除
     *
     * @param index
     */

    fun removeItem(index: Int): T? {
        if (index >= 0 && index < mDatas.size) {
            return mDatas.removeAt(index)
        }
        return null
    }

    fun replaceItem(index: Int,data: T?) {
        if (index >= 0 && index < mDatas.size) {
            if (data != null) {
                mDatas[index] = data
            }
        }
    }

    fun getDatas(): List<T>? {
        return mDatas
    }

    fun clear() {
        mDatas.clear()
    }

    fun showData(list: List<T>?) {
        clear()
        add(list)
        notifyDataSetChanged()
    }

}