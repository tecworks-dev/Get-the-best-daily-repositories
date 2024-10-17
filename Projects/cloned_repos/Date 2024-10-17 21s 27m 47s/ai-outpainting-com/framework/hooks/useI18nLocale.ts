import { DEFAULT_LOCALE } from '@/framework/locale/locale'
import { useMemo } from 'react'
import localeNames from '@/framework/locale/localeConfig'


export type I18nLocales = { key: string, name: string }[]

export default function useI18nLocale(localeKey: string = DEFAULT_LOCALE): [string, I18nLocales] {
  const names = localeNames as Record<string, any>
  const currentLocale = useMemo(() => names[localeKey], [localeKey, names])
  const locales: I18nLocales = useMemo(() =>
    Object.keys(names)
      .filter(key => key !== localeKey)
      .map(key => ({
        key,
        name: names[key]
      })), [names, localeKey])
  return [currentLocale, locales]
}