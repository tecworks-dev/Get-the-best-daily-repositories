import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const tag = await prisma.tag.findUnique({
      where: {
        id,
      },
    });

    if (!tag) {
      return new NextResponse("Tag not found", { status: 404 });
    }

    return NextResponse.json(tag);
  } catch (error) {
    console.error("Error fetching tag:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}

export async function PUT(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const tag = await prisma.tag.findUnique({
      where: {
        id,
      },
    });

    if (!tag) {
      return new NextResponse("Tag not found", { status: 404 });
    }

    const json = await request.json();
    const { name, color } = json;

    // Check if another tag with the same name exists
    if (name && name !== tag.name) {
      const existingTag = await prisma.tag.findFirst({
        where: {
          name,
          id: { not: id }, // Exclude current tag
        },
      });

      if (existingTag) {
        return new NextResponse("Tag with this name already exists", {
          status: 400,
        });
      }
    }

    const updatedTag = await prisma.tag.update({
      where: {
        id,
      },
      data: {
        ...(name && { name }),
        ...(color && { color }),
      },
    });

    return NextResponse.json(updatedTag);
  } catch (error) {
    console.error("Error updating tag:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const tag = await prisma.tag.findUnique({
      where: {
        id,
      },
    });

    if (!tag) {
      return new NextResponse("Tag not found", { status: 404 });
    }

    await prisma.tag.delete({
      where: {
        id,
      },
    });

    return new NextResponse(null, { status: 204 });
  } catch (error) {
    console.error("Error deleting tag:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}
