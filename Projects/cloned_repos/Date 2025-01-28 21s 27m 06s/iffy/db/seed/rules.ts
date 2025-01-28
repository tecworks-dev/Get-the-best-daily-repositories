import { findOrCreateDefaultRuleset } from "@/services/ruleset";
import db from "@/db";
import * as schema from "@/db/schema";

export async function seedRules(clerkOrganizationId: string) {
  const ruleset = await findOrCreateDefaultRuleset(clerkOrganizationId);
  const presets = await db.select().from(schema.presets);

  await db.insert(schema.rules).values(
    presets.map((preset) => ({
      clerkOrganizationId,
      rulesetId: ruleset.id,
      presetId: preset.id,
    })),
  );

  console.log("Seeded Rules");
}
