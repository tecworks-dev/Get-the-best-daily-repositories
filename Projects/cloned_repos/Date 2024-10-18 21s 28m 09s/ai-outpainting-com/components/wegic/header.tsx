'use client'
import React from 'react'
import { t } from '@lingui/macro'
import Navbar from './navbar'

export default function Header({ lang }: { lang?: string }) {
  const navigation = [
    { title: t`Home`, href: '/', current: false },
    { title: t`Pricing`, href: '/pricing', current: false },
    { title: t`Blogs`, href: '/blogs', current: false }
    /*{ name: t`Explore`, href: '/user-case' }*/
  ]
  return <Navbar items={navigation as any} locale={lang} />
}