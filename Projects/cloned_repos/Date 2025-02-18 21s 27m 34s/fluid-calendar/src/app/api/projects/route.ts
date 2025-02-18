import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { ProjectStatus } from "@/types/project";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.getAll("status") as ProjectStatus[];
    const search = searchParams.get("search");

    const projects = await prisma.project.findMany({
      where: {
        ...(status.length > 0 && { status: { in: status } }),
        ...(search && {
          OR: [
            { name: { contains: search } },
            { description: { contains: search } },
          ],
        }),
      },
      include: {
        _count: {
          select: { tasks: true },
        },
      },
      orderBy: {
        createdAt: "desc",
      },
    });

    return NextResponse.json(projects);
  } catch (error) {
    console.error("Error fetching projects:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const json = await request.json();
    const project = await prisma.project.create({
      data: {
        name: json.name,
        description: json.description,
        color: json.color,
        status: json.status || ProjectStatus.ACTIVE,
      },
      include: {
        _count: {
          select: { tasks: true },
        },
      },
    });

    return NextResponse.json(project);
  } catch (error) {
    console.error("Error creating project:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}
