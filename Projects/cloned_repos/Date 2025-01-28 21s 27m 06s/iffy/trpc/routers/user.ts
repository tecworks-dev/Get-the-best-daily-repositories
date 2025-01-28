import db from "@/db";
import protectedProcedure from "../procedures/protected";
import { router } from "../trpc";
import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { eq, inArray, ilike, asc, desc, and, or, lt, gt, sql } from "drizzle-orm";
import * as schema from "@/db/schema";

const paginationSchema = z.object({
  clerkOrganizationId: z.string(),
  cursor: z.object({ sort: z.number().int().optional(), skip: z.number().int().optional() }).default({}),
  limit: z.union([z.literal(10), z.literal(20), z.literal(30), z.literal(40), z.literal(50)]).default(20),
  sorting: z.array(z.object({ id: z.string(), desc: z.boolean() })).default([{ id: "sort", desc: true }]),
  search: z.string().optional(),
  statuses: z.enum(schema.recordUserActions.status.enumValues).array().optional(),
});

const getWhereInput = (
  input: z.infer<typeof paginationSchema>,
  clerkOrganizationId: string,
  cursorValue?: number,
  sortingOrder?: boolean,
) => {
  const { search, statuses } = input;

  return (recordUsers: typeof schema.recordUsers) => {
    const conditions = [eq(recordUsers.clerkOrganizationId, clerkOrganizationId)];

    if (statuses?.length) {
      conditions.push(inArray(recordUsers.actionStatus, statuses));
    }

    if (search) {
      conditions.push(
        or(
          ilike(recordUsers.username, `%${search}%`),
          ilike(recordUsers.email, `%${search}%`),
          ilike(recordUsers.name, `%${search}%`),
          ilike(recordUsers.clientId, `%${search}%`),
        ) ?? sql`true`,
      );
    }

    if (cursorValue !== undefined && sortingOrder !== undefined) {
      const cursorSort = cursorValue;
      if (sortingOrder) {
        conditions.push(lt(recordUsers.sort, cursorSort));
      } else {
        conditions.push(gt(recordUsers.sort, cursorSort));
      }
    }

    return and(...conditions);
  };
};

export const userRouter = router({
  infinite: protectedProcedure.input(paginationSchema).query(async ({ input, ctx }) => {
    const { cursor, limit, sorting } = input;
    const { clerkOrganizationId } = ctx;

    if (clerkOrganizationId !== input.clerkOrganizationId) {
      throw new TRPCError({ code: "UNAUTHORIZED" });
    }

    const supportsCursorPagination = sorting.length === 1 && sorting[0]?.id === "sort";
    let users;
    let nextCursor;

    // Cursor pagination is more performant, but only works with a single sort field
    if (supportsCursorPagination) {
      const { sort } = cursor;
      const sortingOrder = sorting[0]?.desc;
      const orderBy = sortingOrder ? desc(schema.recordUsers.sort) : asc(schema.recordUsers.sort);
      const where = getWhereInput(input, clerkOrganizationId, sort, sortingOrder);

      users = await db.query.recordUsers.findMany({
        where: where(schema.recordUsers),
        limit: limit + 1,
        orderBy: [orderBy],
        with: {
          actions: {
            orderBy: [desc(schema.recordUserActions.createdAt)],
            limit: 1,
          },
        },
      });

      if (users.length > limit) {
        const nextItem = users.pop();
        nextCursor = { sort: nextItem!.sort };
      } else {
        nextCursor = undefined;
      }

      return { users, nextCursor };
    }

    // Offset pagination is more flexible, but less performant
    const { skip } = cursor;
    const offsetValue = skip ?? 0;
    const where = getWhereInput(input, clerkOrganizationId);

    users = await db.query.recordUsers.findMany({
      where: where(schema.recordUsers),
      limit: limit + 1,
      offset: offsetValue,
      orderBy: (usersTable, { asc, desc }) =>
        sorting
          .map(({ id, desc: isDesc }) =>
            id === "name"
              ? isDesc
                ? [desc(usersTable.name), desc(usersTable.email)]
                : [asc(usersTable.name), asc(usersTable.email)]
              : isDesc
                ? [desc(usersTable[id as keyof typeof usersTable])]
                : [asc(usersTable[id as keyof typeof usersTable])],
          )
          .flat(),
      with: {
        actions: {
          orderBy: [desc(schema.recordUserActions.createdAt)],
          limit: 1,
        },
      },
    });

    if (users.length > limit) {
      users.pop();
      nextCursor = { skip: offsetValue + limit };
    } else {
      nextCursor = undefined;
    }

    return { users, nextCursor };
  }),
});
