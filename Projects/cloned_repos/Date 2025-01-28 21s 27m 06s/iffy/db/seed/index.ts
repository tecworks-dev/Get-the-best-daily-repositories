import { findOrCreateDefaultRuleset } from "@/services/ruleset";
import { seedAppeals } from "./appeals";
import { seedRules } from "./rules";
import { seedRecordUsers } from "./record-users";
import { seedRecords } from "./records";
import { seedOrganizationSettings } from "./organization";
import { env } from "@/lib/env";
import { close } from "@/db";
import { seedRecordUserActions } from "./record-user-actions";

async function main() {
  if (!env.SEED_CLERK_ORGANIZATION_ID) {
    console.error("SEED_CLERK_ORGANIZATION_ID is not set");
    process.exit(1);
  }

  const clerkOrganizationId = env.SEED_CLERK_ORGANIZATION_ID;
  await seedOrganizationSettings(clerkOrganizationId);
  const defaultRuleset = await findOrCreateDefaultRuleset(clerkOrganizationId);
  await seedRules(clerkOrganizationId);
  const recordUsers = await seedRecordUsers(clerkOrganizationId);
  await seedRecords(clerkOrganizationId, defaultRuleset, recordUsers);
  await seedRecordUserActions(clerkOrganizationId);
  await seedAppeals(clerkOrganizationId);
}

main()
  .then(() => {
    console.log("Seeding completed successfully.");
    close();
  })
  .catch((e) => {
    console.error(e);
    close();
    process.exit(1);
  });
