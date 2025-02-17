import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { getGoogleCalendarClient } from "@/lib/google-calendar";
import { GaxiosError } from "gaxios";

// Get available (unconnected) calendars for an account
export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const accountId = url.searchParams.get("accountId");

    if (!accountId) {
      return NextResponse.json(
        { error: "Account ID is required" },
        { status: 400 }
      );
    }

    // Get the account
    const account = await prisma.connectedAccount.findUnique({
      where: { id: accountId },
      include: {
        calendars: true,
      },
    });

    if (!account || account.provider !== "GOOGLE") {
      return NextResponse.json({ error: "Invalid account" }, { status: 400 });
    }

    // Create calendar client
    const calendar = await getGoogleCalendarClient(accountId);

    // Get list of calendars
    const calendarList = await calendar.calendarList.list();
    const availableCalendars = calendarList.data.items
      ?.filter((cal) => {
        // Only include calendars that:
        // 1. Have an ID and name
        // 2. Are not already connected
        // 3. User has write access
        return (
          cal.id &&
          cal.summary &&
          !account.calendars.some((f) => f.url === cal.id)
        );
      })
      .map((cal) => ({
        id: cal.id,
        name: cal.summary,
        color: cal.backgroundColor,
        accessRole: cal.accessRole,
      }));

    return NextResponse.json(availableCalendars || []);
  } catch (error) {
    console.error("Failed to list available calendars:", error);
    if (error instanceof GaxiosError && Number(error.code) === 401) {
      return NextResponse.json(
        { error: "Authentication failed. Please try signing in again." },
        { status: 401 }
      );
    }
    return NextResponse.json(
      { error: "Failed to list calendars" },
      { status: 500 }
    );
  }
}
