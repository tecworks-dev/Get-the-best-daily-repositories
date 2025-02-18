import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

// Get a specific feed
export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const feed = await prisma.calendarFeed.findUnique({
      where: { id },
      include: { events: true },
    });

    if (!feed) {
      return NextResponse.json({ error: "Feed not found" }, { status: 404 });
    }

    return NextResponse.json(feed);
  } catch (error) {
    console.error("Failed to fetch feed:", error);
    return NextResponse.json(
      { error: "Failed to fetch feed" },
      { status: 500 }
    );
  }
}

// Update a specific feed
export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const updates = await request.json();
    const updated = await prisma.calendarFeed.update({
      where: { id },
      data: updates,
    });
    return NextResponse.json(updated);
  } catch (error) {
    console.error("Failed to update feed:", error);
    return NextResponse.json(
      { error: "Failed to update feed" },
      { status: 500 }
    );
  }
}

// Delete a specific feed
export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    // The feed's events will be automatically deleted due to the cascade delete in the schema
    await prisma.calendarFeed.delete({
      where: { id },
    });
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete feed:", error);
    return NextResponse.json(
      { error: "Failed to delete feed" },
      { status: 500 }
    );
  }
}
