package com.lyihub.privacy_radar.util

import android.os.Handler
import android.os.Looper
import android.os.Message


class MainHandler: Handler(Looper.getMainLooper()) {
    companion object {
        private var sMainHandler: MainHandler? = null

        fun get(): MainHandler {
            if (sMainHandler == null) {
                sMainHandler = MainHandler()
            }
            return sMainHandler!!
        }
    }

    private val mainHandlers: LinkedHashSet<OnMainHandlerImpl> = LinkedHashSet()


    override fun handleMessage(msg: Message) {
        super.handleMessage(msg)

        for (onMainHandlerImpl in mainHandlers) {
            onMainHandlerImpl.handleMainMessage(msg)
        }
    }

    fun register(onMainHandlerImpl: OnMainHandlerImpl?): MainHandler? {
        if (onMainHandlerImpl != null) {
            mainHandlers.add(onMainHandlerImpl)
        }
        return this
    }

    fun unregister(onMainHandlerImpl: OnMainHandlerImpl?) {
        if (onMainHandlerImpl != null) {
            mainHandlers.remove(onMainHandlerImpl)
        }
    }

    fun clear() {
        mainHandlers.clear()
        if (sMainHandler != null) {
            sMainHandler!!.removeCallbacksAndMessages(null)
        }
    }

    /**
     * 主线程执行
     *
     * @param runnable
     */
    fun runMainThread(runnable: Runnable) {
        get().post(runnable)
    }

    interface OnMainHandlerImpl {
        fun handleMainMessage(message: Message?)
    }
}