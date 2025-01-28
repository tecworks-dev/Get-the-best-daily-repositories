import db from "@/db";
import { inngest } from "@/inngest/client";
import * as schema from "@/db/schema";
import { eq, desc, and } from "drizzle-orm";
import { ViaWithClerkUserOrRecordUser } from "@/lib/types";

type ActionStatus = (typeof schema.recordUserActionStatus.enumValues)[number];

export async function createUserAction({
  clerkOrganizationId,
  recordUserId,
  status,
  via,
  clerkUserId,
}: {
  clerkOrganizationId: string;
  recordUserId: string;
  status: ActionStatus;
} & ViaWithClerkUserOrRecordUser) {
  const recordUser = await db.query.recordUsers.findFirst({
    where: and(
      eq(schema.recordUsers.clerkOrganizationId, clerkOrganizationId),
      eq(schema.recordUsers.id, recordUserId),
    ),
    columns: {
      protected: true,
    },
  });

  if (!recordUser) {
    throw new Error("Record user not found");
  }

  if (recordUser.protected && status !== "Compliant") {
    throw new Error("Record user is protected");
  }

  const lastAction = await db.query.recordUserActions.findFirst({
    where: and(
      eq(schema.recordUserActions.clerkOrganizationId, clerkOrganizationId),
      eq(schema.recordUserActions.recordUserId, recordUserId),
    ),
    orderBy: desc(schema.recordUserActions.createdAt),
    columns: {
      status: true,
    },
  });

  // read the last status
  const lastStatus = lastAction?.status;

  if (lastStatus === status) {
    return lastAction;
  }

  const [userAction] = await db
    .insert(schema.recordUserActions)
    .values({
      clerkOrganizationId,
      status,
      recordUserId,
      via,
      clerkUserId,
    })
    .returning();

  if (!userAction) {
    throw new Error("Failed to create user action");
  }

  // sync the record user status with the new status
  await db
    .update(schema.recordUsers)
    .set({
      actionStatus: status,
      actionStatusCreatedAt: userAction.createdAt,
    })
    .where(
      and(eq(schema.recordUsers.clerkOrganizationId, clerkOrganizationId), eq(schema.recordUsers.id, recordUserId)),
    );

  if (status !== lastStatus) {
    try {
      await inngest.send({
        name: "user-action/status-changed",
        data: {
          clerkOrganizationId,
          id: userAction.id,
          recordUserId,
          status,
          lastStatus: lastStatus ?? null,
        },
      });
    } catch (error) {
      console.error(error);
    }
  }

  return userAction;
}
