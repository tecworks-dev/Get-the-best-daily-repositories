package com.lyihub.privacy_radar.util

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File


object FileUtil {
    const val TAG = "FileUtil"
    fun getSdcardFiles(callback: (List<File>) -> Unit) {
        try {
            val listOfAllFiles = ArrayList<File>()
            CoroutineScope(Dispatchers.IO).launch {

                val directory = File("/sdcard")
                val files = directory.listFiles()
                listOfAllFiles.addAll(files)

                CoroutineScope(Dispatchers.Main).launch {
                    callback.invoke(listOfAllFiles)
                    Log.e(TAG,"getSdcardFiles-count = ${listOfAllFiles.size}")
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}