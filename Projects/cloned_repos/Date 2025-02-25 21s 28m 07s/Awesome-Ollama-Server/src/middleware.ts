import createMiddleware from 'next-intl/middleware';
import { defaultLocale, locales } from './config';

export default createMiddleware({
  defaultLocale,
  locales,
  localePrefix: 'always'
});

export const config = {
  matcher: ['/((?!api|_next|.*\\..*).*)']
}; 