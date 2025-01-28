ALTER TABLE "rule_strategies" ALTER COLUMN "clerk_organization_id" SET NOT NULL;--> statement-breakpoint
ALTER TABLE "preset_strategies" DROP COLUMN IF EXISTS "clerk_organization_id";