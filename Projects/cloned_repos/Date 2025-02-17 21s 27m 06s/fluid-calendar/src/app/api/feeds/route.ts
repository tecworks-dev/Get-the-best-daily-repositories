import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

interface CalendarFeedUpdate {
  id: string;
  enabled?: boolean;
  color?: string | null;
}

// List all calendar feeds
export async function GET() {
  try {
    const feeds = await prisma.calendarFeed.findMany({
      include: {
        account: {
          select: {
            id: true,
            provider: true,
            email: true,
          },
        },
      },
      orderBy: {
        createdAt: "asc",
      },
    });

    return NextResponse.json(feeds);
  } catch (error) {
    console.error("Failed to fetch calendar feeds:", error);
    return NextResponse.json(
      { error: "Failed to fetch calendar feeds" },
      { status: 500 }
    );
  }
}

// Create a new feed
export async function POST(request: Request) {
  try {
    const feed = await request.json();
    const created = await prisma.calendarFeed.create({
      data: feed,
    });
    return NextResponse.json(created);
  } catch (error) {
    console.error("Failed to create feed:", error);
    return NextResponse.json(
      { error: "Failed to create feed" },
      { status: 500 }
    );
  }
}

// Batch update feeds
export async function PUT(request: Request) {
  try {
    const { feeds } = await request.json();

    // Use transaction to ensure all updates succeed or none do
    await prisma.$transaction(
      feeds.map((feed: CalendarFeedUpdate) =>
        prisma.calendarFeed.update({
          where: { id: feed.id },
          data: feed,
        })
      )
    );

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to update feeds:", error);
    return NextResponse.json(
      { error: "Failed to update feeds" },
      { status: 500 }
    );
  }
}

// Update calendar feed settings
export async function PATCH(request: Request) {
  try {
    const { id, enabled, color } = await request.json();

    if (!id) {
      return NextResponse.json(
        { error: "Feed ID is required" },
        { status: 400 }
      );
    }

    const feed = await prisma.calendarFeed.update({
      where: { id },
      data: {
        enabled: enabled !== undefined ? enabled : undefined,
        color: color !== undefined ? color : undefined,
      },
    });

    return NextResponse.json(feed);
  } catch (error) {
    console.error("Failed to update calendar feed:", error);
    return NextResponse.json(
      { error: "Failed to update calendar feed" },
      { status: 500 }
    );
  }
}

// Delete calendar feed
export async function DELETE(request: Request) {
  try {
    const { id } = await request.json();

    if (!id) {
      return NextResponse.json(
        { error: "Feed ID is required" },
        { status: 400 }
      );
    }

    await prisma.calendarFeed.delete({
      where: { id },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete calendar feed:", error);
    return NextResponse.json(
      { error: "Failed to delete calendar feed" },
      { status: 500 }
    );
  }
}
