import { inngest } from "@/inngest/client";
import db from "@/db";
import * as schema from "@/db/schema";
import { getFlaggedRecordsFromUser } from "@/services/users";
import { createUserAction } from "@/services/user-actions";
import { and, eq } from "drizzle-orm/expressions";

const updateUserAfterDeletion = inngest.createFunction(
  { id: "update-user-after-deletion" },
  { event: "record/deleted" },
  async ({ event, step }) => {
    const { clerkOrganizationId, id } = event.data;

    const record = await step.run("fetch-record", async () => {
      const record = await db.query.records.findFirst({
        where: and(eq(schema.records.clerkOrganizationId, clerkOrganizationId), eq(schema.records.id, id)),
        with: {
          recordUser: true,
        },
      });
      if (!record) throw new Error(`Record not found: ${id}`);
      return record;
    });

    const recordUser = record.recordUser;
    if (!recordUser) {
      return;
    }

    const flaggedRecords = await step.run("fetch-user-flagged-records", async () => {
      return await getFlaggedRecordsFromUser({
        clerkOrganizationId,
        id: recordUser.id,
      });
    });

    if (flaggedRecords.length === 0 && recordUser.actionStatus === "Suspended") {
      await step.run("create-user-action", async () => {
        return await createUserAction({
          clerkOrganizationId,
          recordUserId: recordUser.id,
          status: "Compliant",
          via: "Automation",
        });
      });
    }
  },
);

export default [updateUserAfterDeletion];
