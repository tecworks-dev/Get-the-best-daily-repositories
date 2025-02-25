import { getRequestConfig } from 'next-intl/server';
import { defaultLocale } from '../config';

export default getRequestConfig(async ({ locale }) => {
  return {
    messages: (await import(`./locales/${locale || defaultLocale}.json`)).default,
    timeZone: 'UTC',
    now: new Date()
  };
}); 