"use server";

import { z } from "zod";
import { actionClient } from "@/lib/action-client";
import { revalidatePath } from "next/cache";
import * as service from "@/services/user-actions";
import * as schema from "@/db/schema";
import db from "@/db";
import { and, eq, inArray } from "drizzle-orm";

const createUserActionSchema = z.object({
  status: z.enum(schema.recordUserActionStatus.enumValues),
  reasoning: z.string().optional(),
});

// TODO(s3ththompson): Add bulk services in the future
export const createUserActions = actionClient
  .schema(createUserActionSchema)
  .bindArgsSchemas<[recordUserIds: z.ZodArray<z.ZodString>]>([z.array(z.string())])
  .action(
    async ({
      parsedInput: { status, reasoning },
      bindArgsParsedInputs: [recordUserIds],
      ctx: { clerkOrganizationId, clerkUserId },
    }) => {
      const userActions = await Promise.all(
        recordUserIds.map((recordUserId) =>
          service.createUserAction({
            clerkOrganizationId,
            recordUserId,
            status,
            via: "Manual",
            clerkUserId,
          }),
        ),
      );
      for (const recordUserId of recordUserIds) {
        revalidatePath(`/dashboard/users/${recordUserId}`);
      }
      return userActions;
    },
  );

const setUserProtectedSchema = z.boolean();

export const setUserProtectedMany = actionClient
  .schema(setUserProtectedSchema)
  .bindArgsSchemas<[recordUserIds: z.ZodArray<z.ZodString>]>([z.array(z.string())])
  .action(async ({ parsedInput, bindArgsParsedInputs: [recordUserIds], ctx: { clerkOrganizationId } }) => {
    const userRecords = await db
      .update(schema.recordUsers)
      .set({
        protected: parsedInput,
      })
      .where(
        and(
          eq(schema.recordUsers.clerkOrganizationId, clerkOrganizationId),
          inArray(schema.recordUsers.id, recordUserIds),
        ),
      )
      .returning();

    for (const recordUserId of recordUserIds) {
      revalidatePath(`/dashboard/users/${recordUserId}`);
    }
    return userRecords;
  });
