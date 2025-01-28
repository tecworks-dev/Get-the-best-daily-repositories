"use server";

import { z } from "zod";
import { actionClient } from "@/lib/action-client";
import * as schema from "@/db/schema";
import * as service from "@/services/moderations";
import { revalidatePath } from "next/cache";
import { inngest } from "@/inngest/client";

const createModerationSchema = z.object({
  status: z.enum(schema.moderationStatus.enumValues),
  reasoning: z.string().optional(),
});

// TODO(s3ththompson): Add bulk services in the future
export const createModerations = actionClient
  .schema(createModerationSchema)
  .bindArgsSchemas<[recordIds: z.ZodArray<z.ZodString>]>([z.array(z.string())])
  .action(
    async ({
      parsedInput: { status, reasoning },
      bindArgsParsedInputs: [recordIds],
      ctx: { clerkOrganizationId, clerkUserId },
    }) => {
      const moderations = await Promise.all(
        recordIds.map((recordId) =>
          service.createModeration({
            clerkOrganizationId,
            recordId,
            status,
            reasoning,
            via: "Manual",
            clerkUserId,
          }),
        ),
      );
      for (const recordId of recordIds) {
        revalidatePath(`/dashboard/records/${recordId}`);
      }
      return moderations;
    },
  );

// TODO(s3ththompson): Add bulk services in the future
export const moderateMany = actionClient
  .bindArgsSchemas<[recordIds: z.ZodArray<z.ZodString>]>([z.array(z.string())])
  .action(async ({ bindArgsParsedInputs: [recordIds], ctx: { clerkOrganizationId } }) => {
    const pendingModerations = await Promise.all(
      recordIds.map(async (recordId) => {
        const pendingModeration = await service.createPendingModeration({
          clerkOrganizationId,
          recordId,
          via: "AI",
        });
        try {
          await inngest.send({
            name: "moderation/moderated",
            data: {
              clerkOrganizationId,
              moderationId: pendingModeration.id,
              recordId,
            },
          });
        } catch (error) {
          console.error(error);
        }
        return pendingModeration;
      }),
    );
    for (const recordId of recordIds) {
      revalidatePath(`/dashboard/records/${recordId}`);
    }
    return pendingModerations;
  });
