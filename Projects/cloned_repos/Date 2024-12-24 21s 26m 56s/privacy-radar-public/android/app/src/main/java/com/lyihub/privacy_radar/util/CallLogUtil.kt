package com.lyihub.privacy_radar.util

import android.content.Context
import android.provider.CallLog
import android.provider.ContactsContract
import android.provider.MediaStore
import android.util.Log
import com.lyihub.privacy_radar.data.CallLogInfo
import com.lyihub.privacy_radar.data.ContactInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch


object CallLogUtil {
    const val TAG = "CallLogUtil"
    fun getAllCallLog(context: Context, callback: (List<CallLogInfo>) -> Unit) {
        try {
            val listOfAllContacts = ArrayList<CallLogInfo>()
            CoroutineScope(Dispatchers.IO).launch {
                val uri = CallLog.Calls.CONTENT_URI
                val projection = arrayOf(
                    CallLog.Calls.CACHED_NAME,
                    CallLog.Calls.NUMBER
                )
                val cursor = context.contentResolver.query(uri, projection, null, null, null)

                val nameIndex = cursor?.getColumnIndex(CallLog.Calls.CACHED_NAME) ?: 0
                var phoneIndex = cursor?.getColumnIndex(CallLog.Calls.NUMBER) ?: 0
                while (cursor?.moveToNext() == true) {
                    val name = cursor.getString(nameIndex) ?: "未备注"
                    var phone = cursor.getString(phoneIndex)
                    Log.e(TAG,"getAllCallLog-name = $name")
                    Log.e(TAG,"getAllCallLog-phone = $phone")

                    val item = CallLogInfo()
                    item.name = name
                    item.phone = phone
                    listOfAllContacts.add(item)
                }
                cursor?.close()

                CoroutineScope(Dispatchers.Main).launch {
                    callback.invoke(listOfAllContacts)
                    Log.e(TAG,"getAllContacts-count = ${listOfAllContacts.size}")
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}