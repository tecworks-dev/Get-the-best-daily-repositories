import { renderEmailTemplate } from "@/services/email";

import { eq } from "drizzle-orm";
import db from "../index";
import * as schema from "../schema";
import sample from "lodash/sample";
import { createMessage } from "@/services/messages";

export async function seedRecordUserActions(clerkOrganizationId: string) {
  const recordUsers = await db.query.recordUsers.findMany({
    where: eq(schema.recordUsers.clerkOrganizationId, clerkOrganizationId),
    with: {
      records: true,
    },
  });

  const recordUserActions = await db
    .insert(schema.recordUserActions)
    .values(
      recordUsers.map((recordUser) => {
        const isFlagged = recordUser.records.some((record) => record.moderationStatus === "Flagged");
        const status = isFlagged && !recordUser.protected ? sample(["Suspended", "Banned"] as const) : "Compliant";
        return {
          clerkOrganizationId,
          recordUserId: recordUser.id,
          status,
          createdAt: recordUser.createdAt,
        } as const;
      }),
    )
    .returning();

  const { subject, body } = await renderEmailTemplate({
    clerkOrganizationId,
    type: "Suspended",
  });

  for (const recordUserAction of recordUserActions) {
    await db
      .update(schema.recordUsers)
      .set({
        actionStatus: recordUserAction.status,
        actionStatusCreatedAt: recordUserAction.createdAt,
      })
      .where(eq(schema.recordUsers.id, recordUserAction.recordUserId));

    if (recordUserAction.status === "Suspended") {
      await createMessage({
        clerkOrganizationId,
        userActionId: recordUserAction.id,
        type: "Outbound",
        toId: recordUserAction.recordUserId,
        subject,
        text: body,
      });
    }
  }

  console.log("Seeded Record User Actions");

  return recordUserActions;
}
