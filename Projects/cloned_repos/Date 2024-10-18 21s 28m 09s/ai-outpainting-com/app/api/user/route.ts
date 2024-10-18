import { getCurrentUser } from "@/auth"
import { NextResponse } from "next/server"

export async function GET(req: Request) {
  const user = await getCurrentUser()
  return NextResponse.json(user)
}
