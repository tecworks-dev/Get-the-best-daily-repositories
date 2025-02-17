import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET() {
  try {
    const tags = await prisma.tag.findMany({
      orderBy: {
        name: "asc",
      },
    });

    return NextResponse.json(tags);
  } catch (error) {
    console.error("Error fetching tags:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    console.log("Received tag creation request with body:", body);

    if (!body || typeof body.name !== "string" || !body.name.trim()) {
      console.log("Tag validation failed:", {
        hasBody: !!body,
        nameType: typeof body?.name,
        nameTrimmed: body?.name?.trim?.(),
      });
      return new NextResponse(
        JSON.stringify({
          error: "Name is required",
          details: {
            hasBody: !!body,
            nameType: typeof body?.name,
            receivedName: body?.name,
          },
        }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    const name = body.name.trim();
    const color = body.color;

    // Check if tag with same name already exists
    const existingTag = await prisma.tag.findFirst({
      where: {
        name,
      },
    });

    if (existingTag) {
      return new NextResponse(
        JSON.stringify({ error: "Tag with this name already exists" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const tag = await prisma.tag.create({
      data: {
        name,
        color,
      },
    });

    return NextResponse.json(tag);
  } catch (error) {
    console.error("Error creating tag:", error);
    return new NextResponse(
      JSON.stringify({ error: "Internal Server Error" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
