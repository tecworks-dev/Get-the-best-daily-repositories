"use server";

import db from "@/db";
import * as schema from "@/db/schema";
import { eq, and } from "drizzle-orm";
import crypto from "crypto";
import { decrypt, encrypt } from "./encrypt";

export type WebhookEvents = {
  "record.flagged": {
    entity: string;
    clientId: string;
    user?: {
      protected: true;
    };
  };
  "record.compliant": {
    entity: string;
    clientId: string;
    user?: {
      protected: true;
    };
  };
  "user.suspended": {
    clientId: string;
  };
  "user.compliant": {
    clientId: string;
  };
  "user.banned": {
    clientId: string;
  };
};

export async function createWebhook({ clerkOrganizationId, url }: { clerkOrganizationId: string; url: string }) {
  const secret = crypto.randomBytes(32).toString("hex");
  const [webhook] = await db
    .insert(schema.webhookEndpoints)
    .values({
      clerkOrganizationId,
      url,
      secret: encrypt(secret),
    })
    .returning();

  if (!webhook) {
    throw new Error("Failed to create webhook");
  }

  webhook.secret = decrypt(webhook.secret);
  return webhook;
}

export async function updateWebhookUrl({
  clerkOrganizationId,
  id,
  url,
}: {
  clerkOrganizationId: string;
  id: string;
  url: string;
}) {
  const [updatedWebhook] = await db
    .update(schema.webhookEndpoints)
    .set({
      url,
    })
    .where(
      and(eq(schema.webhookEndpoints.id, id), eq(schema.webhookEndpoints.clerkOrganizationId, clerkOrganizationId)),
    )
    .returning();

  if (!updatedWebhook) {
    throw new Error("Webhook not found or not authorized to update");
  }

  return updatedWebhook;
}

export async function sendWebhook<T extends keyof WebhookEvents>({
  id,
  event,
  payload,
}: {
  id: string;
  event: T;
  payload: WebhookEvents[T];
}) {
  const webhook = await db.query.webhookEndpoints.findFirst({
    where: eq(schema.webhookEndpoints.id, id),
  });

  if (!webhook) {
    throw new Error("Webhook not found");
  }

  webhook.secret = decrypt(webhook.secret);

  const timestamp = Date.now().toString();
  const body = JSON.stringify({ event, payload, timestamp });
  const signature = crypto.createHmac("sha256", webhook.secret).update(body).digest("hex");

  try {
    const response = await fetch(webhook.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        "X-Signature": signature,
      },
      body,
    });

    if (!response.ok) {
      throw new Error(`Error sending webhook: ${response.statusText}`);
    }
  } catch (error) {
    console.error("Error sending webhook:", error);
    throw error;
  }
}
