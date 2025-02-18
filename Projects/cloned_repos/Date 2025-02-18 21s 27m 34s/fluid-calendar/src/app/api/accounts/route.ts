import { NextResponse } from "next/server";
import { TokenManager } from "@/lib/token-manager";

export async function GET() {
  try {
    const tokenManager = TokenManager.getInstance();
    const accounts = await tokenManager.listAccounts();
    return NextResponse.json(accounts);
  } catch (error) {
    console.error("Failed to list accounts:", error);
    return NextResponse.json(
      { error: "Failed to list accounts" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const { accountId } = await request.json();
    if (!accountId) {
      return NextResponse.json(
        { error: "Account ID is required" },
        { status: 400 }
      );
    }

    const tokenManager = TokenManager.getInstance();
    await tokenManager.removeAccount(accountId);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to remove account:", error);
    return NextResponse.json(
      { error: "Failed to remove account" },
      { status: 500 }
    );
  }
}
