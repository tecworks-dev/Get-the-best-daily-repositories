package com.lyihub.privacy_radar.holder

import android.text.TextUtils
import android.util.Log
import android.view.View
import com.lyihub.privacy_radar.data.TestReportInfo
import com.lyihub.privacy_radar.util.ResUtils
import com.lyihub.privacy_radar.util.SpannableUtil
import kotlinx.android.synthetic.main.rv_test_report_cell.view.*


class TestReportContentHolder(itemView: View) : ContentViewHolder(itemView) {

    fun bindData(data: TestReportInfo?) {
        itemView.mTvContent.text = data?.content ?: ""
        if (data?.resultType == 1) {
            itemView.mTvTestResult.text = "成功"
        } else {
            var dotTextSize = ResUtils.getDimenPixRes(com.victor.screen.match.library.R.dimen.dp_24)
            val text = "${data?.result}${data?.resultUnit}"
            val spanText = data?.resultUnit
            if (!TextUtils.isEmpty(text) && !TextUtils.isEmpty(spanText)) {
                itemView.mTvTestResult.text = SpannableUtil.getSpannableTextSize(dotTextSize,text, spanText)
            }
        }
    }

    override fun onLongClick(v: View): Boolean {
        return false
    }
}