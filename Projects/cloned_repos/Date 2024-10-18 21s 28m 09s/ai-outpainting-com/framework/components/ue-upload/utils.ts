import { random } from 'radash'
import dayjs from 'dayjs'
import { UeUploadProps } from './types'
import { sanitizeUrl } from '@/framework/utils'

export const UPLOAD_PUBLIC_DIR = '/public'
const UPLOAD_RANDOM_CHARS = 'QWERTYUIOPASDFGHJKLZXCVBNM1234567890'

export function createUploadFileName(name: string, len = 8) {
  const suffix = name.substring(name.lastIndexOf('.') - 1)
  let n = ''
  for (let i = 0; i < len; i++) {
    n += UPLOAD_RANDOM_CHARS[random(0, UPLOAD_RANDOM_CHARS.length - 1)]
  }
  return suffix ? `${n}.${suffix}` : n
}

export function createUploadFileKey(name: string, props: UeUploadProps, noPrefix = false) {
  let key = props.dir ?? UPLOAD_PUBLIC_DIR
  if (props?.withTimestamp) {
    key += `/${dayjs().format('YYYY-MM-DD')}`
  }
  if (props?.keepOriginName) {
    key += `/${sanitizeUrl(name)}`
  } else {
    key += `/${createUploadFileName(name)}`
  }
  if (noPrefix) {
    key = key.substring(1)
  }
  return key
}
