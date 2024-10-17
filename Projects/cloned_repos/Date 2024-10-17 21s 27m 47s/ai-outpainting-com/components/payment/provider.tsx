'use client'
import { PayPalScriptProvider, ReactPayPalScriptOptions } from '@paypal/react-paypal-js'
import React from 'react'


export function WrappedPaypalProvider({ locale, children }: { locale?: string, children: React.ReactNode }) {
  let options = {
    sdkBaseUrl:'https://www.paypal.com/sdk/js',
    clientId: process.env['UE_PAYPAL_CLIENT_ID'],
    currency: 'USD',
    intent: 'capture'
  } as ReactPayPalScriptOptions
  // 开发环境使用沙盒模式
  if(process.env.NODE_ENV==='development'){
    options.sdkBaseUrl='https://www.sandbox.paypal.com/sdk/js'
  }

  return <PayPalScriptProvider options={options}>
    {children}
  </PayPalScriptProvider>
}