import { siteConfig } from '@/config/site'
import { activateLocale, AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { t } from '@lingui/macro'
import { Metadata } from 'next'
import EditorView from './view'

// 动态生成metadata
export async function generateMetadata({
  params,
}: {
  params: { lang: AVAILABLE_LOCALES }
}): Promise<Metadata> {
  await activateLocale(params.lang)

  const title = t`AI Outpainting Playground` + ` | ${siteConfig.name}`
  return {
    title,
  }
}

export default async function EditorPage({
  params: { lang },
}: {
  params: { lang: AVAILABLE_LOCALES }
}) {
  await activateLocale(lang)
  
  return <EditorView params={{lang}}/>
}


