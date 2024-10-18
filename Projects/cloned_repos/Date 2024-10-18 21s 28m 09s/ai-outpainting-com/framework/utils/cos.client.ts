export function wrapUrl(k: string | null | undefined, cred: any) {
  if (k && k.startsWith('http')) {
    return k ?? ''
  }
  if (
    k &&
    k.startsWith('/') &&
    k.includes('public') &&
    cred &&
    cred.publicPath
  ) {
    const { publicPath } = cred
    return `${publicPath}${k}`
  }
  return k ?? ''
}

export function wrapDownloadUrl(k: string | null | undefined, cred: any) {
  if (k && k.startsWith('http')) {
    return k ?? ''
  }
  if (
    k &&
    k.startsWith('/') &&
    k.includes('public') &&
    cred &&
    cred.publicPath
  ) {
    console.log('k', k)
    const { publicPath } = cred
    const url = `${publicPath}${k}`

    return (
      url +
      (url.includes('?') ? '&' : '?') +
      'response-content-disposition=attachment'
    )
  }
  return k ?? ''
}
