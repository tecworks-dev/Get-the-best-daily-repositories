import { NextRequest } from 'next/server'
import { R } from '@/framework/utils'
import { createGetSingedUrl, createPutSingedUrl } from '@/framework/utils/s34r2'


export async function GET(request: NextRequest) {
  const key = request.nextUrl.searchParams.get('key')
  const op = request.nextUrl.searchParams.get('op')
  if (!key) {
    return R.error('key必填')
  }
  try {
    if ('read' === op) {
      const url = await createGetSingedUrl(key)
      return R.ok(url)
    }
    const url = await createPutSingedUrl(key)
    return R.ok(url)
  } catch (error) {
    console.log('创建预签名链接异常', error)
    return R.error('创建预签名链接异常')
  }
}