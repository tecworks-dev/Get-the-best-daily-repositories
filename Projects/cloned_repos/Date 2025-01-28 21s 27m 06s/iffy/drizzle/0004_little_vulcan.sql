UPDATE "organization_settings" SET "blocklist" = '{}' WHERE "blocklist" IS NULL;--> statement-breakpoint
ALTER TABLE "organization_settings" ALTER COLUMN "blocklist" SET DEFAULT '{}';--> statement-breakpoint
ALTER TABLE "organization_settings" ALTER COLUMN "blocklist" SET NOT NULL;--> statement-breakpoint
UPDATE "ruleset_categories" SET "blocklist" = '{}' WHERE "blocklist" IS NULL;--> statement-breakpoint
ALTER TABLE "ruleset_categories" ALTER COLUMN "blocklist" SET DEFAULT '{}';--> statement-breakpoint
ALTER TABLE "ruleset_categories" ALTER COLUMN "blocklist" SET NOT NULL;