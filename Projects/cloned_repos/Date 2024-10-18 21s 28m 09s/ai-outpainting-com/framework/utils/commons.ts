export function isBoolean(value: any) {
  return (
    value === true ||
    value === false ||
    (isObjectLike(value) && getTag(value) === '[object Boolean]')
  )
}

export function isObjectLike(value: any) {
  return typeof value === 'object' && value !== null
}

export function getTag(value: any) {
  if (value == null) {
    return value === undefined ? '[object Undefined]' : '[object Null]'
  }
  return toString.call(value)
}

export function getBaseUrl(){
  return process.env.UE_WEB_API_URL
}
