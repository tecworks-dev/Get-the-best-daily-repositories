import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

interface CalendarEventInput {
  start: string | Date;
  end: string | Date;
  created?: string | Date;
  lastModified?: string | Date;
  [key: string]: unknown;
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { events } = await request.json();
    const { id: feedId } = await params;

    // Start a transaction to ensure data consistency
    await prisma.$transaction(async (tx) => {
      // Delete existing events for this feed
      await tx.calendarEvent.deleteMany({
        where: { feedId },
      });

      // Insert new events
      if (events && events.length > 0) {
        await tx.calendarEvent.createMany({
          data: events.map((event: CalendarEventInput) => ({
            ...event,
            feedId,
            // Convert Date objects to strings for database storage
            start: new Date(event.start).toISOString(),
            end: new Date(event.end).toISOString(),
            created: event.created
              ? new Date(event.created).toISOString()
              : undefined,
            lastModified: event.lastModified
              ? new Date(event.lastModified).toISOString()
              : undefined,
          })),
        });
      }

      // Update feed's lastSync timestamp
      await tx.calendarFeed.update({
        where: { id: feedId },
        data: { lastSync: new Date() },
      });
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to sync feed:", error);
    return NextResponse.json({ error: "Failed to sync feed" }, { status: 500 });
  }
}
