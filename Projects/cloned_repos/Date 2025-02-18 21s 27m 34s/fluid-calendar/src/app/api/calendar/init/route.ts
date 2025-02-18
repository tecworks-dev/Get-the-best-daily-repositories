import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { v4 as uuidv4 } from "uuid";

export async function POST() {
  try {
    // Check if local calendar exists
    const localCalendar = await prisma.calendarFeed.findFirst({
      where: { type: "LOCAL" },
    });

    // If no local calendar exists, create one
    if (!localCalendar) {
      await prisma.calendarFeed.create({
        data: {
          id: uuidv4(),
          name: "My Calendar",
          type: "LOCAL",
          color: "#3b82f6", // Default blue color
          enabled: true,
        },
      });
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to initialize calendars:", error);
    return NextResponse.json(
      { error: "Failed to initialize calendars" },
      { status: 500 }
    );
  }
}
