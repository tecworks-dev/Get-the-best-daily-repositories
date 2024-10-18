'use server'
import { NextRequest, NextResponse } from 'next/server'
import { getCosCredential } from '@/framework/utils/cos'

export async function GET(request: NextRequest) {
  const data = await getCosCredential()
  return NextResponse.json(data)
}
