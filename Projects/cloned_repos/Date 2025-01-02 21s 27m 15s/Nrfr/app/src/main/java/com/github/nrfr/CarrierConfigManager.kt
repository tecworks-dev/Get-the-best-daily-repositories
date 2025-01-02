package com.github.nrfr

import android.content.Context
import android.os.PersistableBundle
import android.telephony.CarrierConfigManager
import android.telephony.SubscriptionManager
import android.telephony.TelephonyFrameworkInitializer
import android.telephony.TelephonyManager
import com.android.internal.telephony.ICarrierConfigLoader
import rikka.shizuku.ShizukuBinderWrapper

object CarrierConfigManager {
    fun getSimCards(context: Context): List<SimCardInfo> {
        val simCards = mutableListOf<SimCardInfo>()
        val subId1 = SubscriptionManager.getSubId(0)
        val subId2 = SubscriptionManager.getSubId(1)

        if (subId1 != null) {
            val config1 = getCurrentConfig(subId1[0])
            simCards.add(SimCardInfo(1, subId1[0], getCarrierNameBySubId(context, subId1[0]), config1))
        }
        if (subId2 != null) {
            val config2 = getCurrentConfig(subId2[0])
            simCards.add(SimCardInfo(2, subId2[0], getCarrierNameBySubId(context, subId2[0]), config2))
        }

        return simCards
    }

    private fun getCurrentConfig(subId: Int): Map<String, String> {
        try {
            val carrierConfigLoader = ICarrierConfigLoader.Stub.asInterface(
                ShizukuBinderWrapper(
                    TelephonyFrameworkInitializer
                        .getTelephonyServiceManager()
                        .carrierConfigServiceRegisterer
                        .get()
                )
            )
            val config = carrierConfigLoader.getConfigForSubId(subId, "com.github.nrfr") ?: return emptyMap()

            val result = mutableMapOf<String, String>()

            // 获取国家码配置
            config.getString(CarrierConfigManager.KEY_SIM_COUNTRY_ISO_OVERRIDE_STRING)?.let {
                result["国家码"] = it
            }

            // 获取运营商名称配置
            if (config.getBoolean(CarrierConfigManager.KEY_CARRIER_NAME_OVERRIDE_BOOL, false)) {
                config.getString(CarrierConfigManager.KEY_CARRIER_NAME_STRING)?.let {
                    result["运营商名称"] = it
                }
            }

            return result
        } catch (e: Exception) {
            return emptyMap()
        }
    }

    private fun getCarrierNameBySubId(context: Context, subId: Int): String {
        val telephonyManager = context.getSystemService(Context.TELEPHONY_SERVICE) as? TelephonyManager
            ?: return ""
        return telephonyManager.getNetworkOperatorName(subId)
    }

    fun setCarrierConfig(subId: Int, countryCode: String?) {
        val bundle = PersistableBundle()

        if (!countryCode.isNullOrEmpty()) {
            if (countryCode.length == 2) {
                bundle.putString(
                    CarrierConfigManager.KEY_SIM_COUNTRY_ISO_OVERRIDE_STRING,
                    countryCode.lowercase()
                )
            }
        }

        overrideCarrierConfig(subId, bundle)
    }

    fun resetCarrierConfig(subId: Int) {
        overrideCarrierConfig(subId, null)
    }

    private fun overrideCarrierConfig(subId: Int, bundle: PersistableBundle?) {
        val carrierConfigLoader = ICarrierConfigLoader.Stub.asInterface(
            ShizukuBinderWrapper(
                TelephonyFrameworkInitializer
                    .getTelephonyServiceManager()
                    .carrierConfigServiceRegisterer
                    .get()
            )
        )
        carrierConfigLoader.overrideConfig(subId, bundle, true)
    }
}

data class SimCardInfo(
    val slot: Int,
    val subId: Int,
    val carrierName: String,
    val currentConfig: Map<String, String> = emptyMap()
)
