import { useMemo } from 'react'


export default function useNavigation(pathname: string): [string, (href: string) => boolean] {
  const pathWithOutLocale = useMemo(() => `${pathname.replace(/^\/[^\/]*\//, '/')}`, [pathname])

  function isActive(href: string): boolean {
    if (pathWithOutLocale === '/' && (href === '' || href === '/')) {
      return true
    }
    return pathWithOutLocale.startsWith(href) && href !== '/'
  }

  return [pathWithOutLocale, isActive]
}