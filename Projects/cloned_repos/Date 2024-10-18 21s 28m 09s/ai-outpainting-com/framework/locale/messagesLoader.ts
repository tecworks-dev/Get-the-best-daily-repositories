import { Messages } from '@lingui/core'
import { AVAILABLE_LOCALES, DEFAULT_LOCALE, isAcceptedLocale } from './locale'

export const loadTranslationMessagesOnServerSide = async (
  locale: AVAILABLE_LOCALES,
): Promise<Messages | null> => {
  const { messages } = await import(
    `@/translations/${!isAcceptedLocale(locale) ? DEFAULT_LOCALE : locale}/messages.json`
  )
  return messages
}
