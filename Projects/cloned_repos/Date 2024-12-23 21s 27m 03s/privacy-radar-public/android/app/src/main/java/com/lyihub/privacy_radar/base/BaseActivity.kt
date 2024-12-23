package com.lyihub.privacy_radar.base

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.lyihub.privacy_radar.util.StatusBarUtil


abstract class BaseActivity: AppCompatActivity() {
    /**
     * 是否需要使用DataBinding 供子类BaseVmDbActivity修改，用户请慎动
     */
    private var isUserDb = false
    protected var TAG = javaClass.simpleName
    var statusBarTextColorBlack: Boolean = true

    abstract fun getLayoutResource(): Int

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!isUserDb) {
            setContentView(getLayoutResource())
        } else {
            initDataBind()
        }
        //状态栏背景及字体颜色适配
        StatusBarUtil.translucentStatusBar(this, true,statusBarTextColorBlack,true)
    }

    /**
     * 供子类BaseDbActivity 初始化DataBinding操作
     */
    open fun initDataBind() {}

    internal fun userDataBinding(isUserDb: Boolean) {
        this.isUserDb = isUserDb
    }

}