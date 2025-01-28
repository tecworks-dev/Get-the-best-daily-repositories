import db from "@/db";
import { inngest } from "@/inngest/client";
import * as schema from "@/db/schema";
import { eq, and, sql } from "drizzle-orm";
import { findUrlsInText } from "@/services/url-moderation";

export async function createOrUpdateRecord({
  clerkOrganizationId,
  clientId,
  name,
  entity,
  text,
  imageUrls,
  externalUrls,
  clientUrl,
  recordUserId,
  createdAt,
}: {
  clerkOrganizationId: string;
  clientId: string;
  clientUrl?: string;
  name: string;
  entity: string;
  text: string;
  imageUrls?: string[];
  externalUrls?: string[];
  recordUserId?: string;
  createdAt?: Date;
}) {
  const lastRecord = await db.query.records.findFirst({
    where: and(eq(schema.records.clerkOrganizationId, clerkOrganizationId), eq(schema.records.clientId, clientId)),
    columns: {
      recordUserId: true,
    },
  });

  const [record] = await db
    .insert(schema.records)
    .values({
      clerkOrganizationId,
      clientId,
      clientUrl,
      name,
      entity,
      text,
      imageUrls,
      externalUrls,
      recordUserId,
      createdAt,
    })
    .onConflictDoUpdate({
      target: schema.records.clientId,
      set: {
        name,
        entity,
        text,
        imageUrls,
        externalUrls,
        clientUrl,
        recordUserId,
      },
    })
    .returning();

  if (!record) {
    throw new Error("Failed to upsert record");
  }

  if (record.moderationStatus === "Flagged") {
    const userRemoved = !!lastRecord?.recordUserId && !record.recordUserId;
    const userAdded = !lastRecord?.recordUserId && !!record.recordUserId;
    const userChanged =
      !!lastRecord?.recordUserId && !!record.recordUserId && lastRecord.recordUserId !== record.recordUserId;

    if (userRemoved || userChanged) {
      await db
        .update(schema.recordUsers)
        .set({
          flaggedRecordsCount: sql`${schema.recordUsers.flaggedRecordsCount} - 1`,
        })
        .where(
          and(
            eq(schema.recordUsers.clerkOrganizationId, clerkOrganizationId),
            eq(schema.recordUsers.id, lastRecord.recordUserId!),
          ),
        );
    }

    if (userAdded || userChanged) {
      await db
        .update(schema.recordUsers)
        .set({
          flaggedRecordsCount: sql`${schema.recordUsers.flaggedRecordsCount} + 1`,
        })
        .where(
          and(
            eq(schema.recordUsers.clerkOrganizationId, clerkOrganizationId),
            eq(schema.recordUsers.id, record.recordUserId!),
          ),
        );
    }
  }

  return record;
}

export async function deleteRecord(clerkOrganizationId: string, recordId: string) {
  const [record] = await db
    .update(schema.records)
    .set({
      deletedAt: new Date(),
    })
    .where(and(eq(schema.records.clerkOrganizationId, clerkOrganizationId), eq(schema.records.id, recordId)))
    .returning();

  if (!record) {
    throw new Error("Failed to delete record");
  }

  if (record.recordUserId && record.moderationStatus === "Flagged") {
    await db
      .update(schema.recordUsers)
      .set({
        flaggedRecordsCount: sql`${schema.recordUsers.flaggedRecordsCount} - 1`,
      })
      .where(
        and(
          eq(schema.recordUsers.clerkOrganizationId, clerkOrganizationId),
          eq(schema.recordUsers.id, record.recordUserId),
        ),
      );
  }

  try {
    await inngest.send({
      name: "record/deleted",
      data: { clerkOrganizationId, id: recordId },
    });
  } catch (error) {
    console.error(error);
  }

  return record;
}

export function getRecordUrls({ text, externalUrls }: { text: string; externalUrls?: string[] }) {
  const embeddedUrls = findUrlsInText(text);
  const allLinks = Array.from(new Set([...embeddedUrls, ...(externalUrls ?? [])]));
  return allLinks;
}
