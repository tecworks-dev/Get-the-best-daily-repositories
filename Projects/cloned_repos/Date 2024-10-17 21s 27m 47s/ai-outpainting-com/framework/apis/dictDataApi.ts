'use server'
import dictDataService from '@/framework/services/DictDataService'
import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  const dictType = request.nextUrl.searchParams.get('dictType')
  const data = await dictDataService.findByDictType(dictType as string)
  return NextResponse.json(data)
}
