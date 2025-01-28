DROP TABLE "category_openai_moderation_parameters" CASCADE;--> statement-breakpoint
DROP TABLE "_ModerationToRulesetCategory" CASCADE;--> statement-breakpoint
DROP TABLE "openai_moderation_parameters" CASCADE;--> statement-breakpoint
DROP TABLE "ruleset_categories" CASCADE;--> statement-breakpoint
ALTER TABLE "organization_settings" DROP COLUMN IF EXISTS "categories_enabled";--> statement-breakpoint
ALTER TABLE "organization_settings" DROP COLUMN IF EXISTS "blocklist";