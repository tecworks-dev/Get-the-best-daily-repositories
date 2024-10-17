import * as React from 'react'
import AppWithTranslation from '@/framework/locale/AppWithTranslation'
import { NextUIProvider } from '@nextui-org/react'
import { ServerSideGeneratedI18nNamespace } from '@/framework/locale/types'
import withTheme from '@/framework/theme/antdWithTheme'
import { GoogleOAuthProvider } from '@react-oauth/google'
import { SessionProvider } from 'next-auth/react'
import { auth } from '@/auth'
import { GetServerSideProps } from 'next'
import { Session } from '@auth/core/types'
export interface ProvidersProps {
  children: React.ReactNode
  params: { i18n: ServerSideGeneratedI18nNamespace },
}

export const getServerSideProps: GetServerSideProps = async (context) => {
  const session = await auth(context)
  return {
    props: { session }
  }
}

export async function Providers({ children, params }: ProvidersProps) {
  return (
    <GoogleOAuthProvider clientId={process.env.UE_GOOGLE_CLIENT_ID!}>
      <SessionProvider>
        <NextUIProvider locale={params.i18n.locale}>
          <AppWithTranslation i18n={params.i18n}>
              {withTheme(children)}
          </AppWithTranslation>
        </NextUIProvider>
      </SessionProvider>
    </GoogleOAuthProvider>
  )
}
