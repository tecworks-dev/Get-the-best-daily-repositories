package com.lyihub.privacy_radar.data

import java.io.Serializable


class SmsInfo: Serializable {
    var address: String? = null
    var body: String? = null
    var date: Long = 0
}