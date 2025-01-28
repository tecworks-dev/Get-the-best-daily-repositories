import db from "../index";
import * as schema from "../schema";
import { findOrCreateOrganizationSettings } from "@/services/organization-settings";
import { eq } from "drizzle-orm";

export async function seedOrganizationSettings(clerkOrganizationId: string) {
  const organizationSettings = await findOrCreateOrganizationSettings(clerkOrganizationId);
  const [updatedSettings] = await db
    .update(schema.organizationSettings)
    .set({ testModeEnabled: false, emailsEnabled: true })
    .where(eq(schema.organizationSettings.id, organizationSettings.id))
    .returning();
  return [updatedSettings];
}
