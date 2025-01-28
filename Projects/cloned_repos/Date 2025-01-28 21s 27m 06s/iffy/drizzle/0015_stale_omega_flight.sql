CREATE TABLE IF NOT EXISTS "preset_strategies" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text,
	"type" "StrategyType" NOT NULL,
	"preset_id" text NOT NULL,
	"options" jsonb NOT NULL,
	"created_at" timestamp (3) DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "strategies" RENAME TO "rule_strategies";--> statement-breakpoint
ALTER TABLE "rule_strategies" DROP CONSTRAINT "strategies_rule_id_fkey";
--> statement-breakpoint
ALTER TABLE "rule_strategies" DROP CONSTRAINT "strategies_preset_id_fkey";
--> statement-breakpoint
ALTER TABLE "rule_strategies" ALTER COLUMN "rule_id" SET NOT NULL;--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "preset_strategies" ADD CONSTRAINT "preset_strategies_preset_id_fkey" FOREIGN KEY ("preset_id") REFERENCES "public"."presets"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "rule_strategies" ADD CONSTRAINT "rule_strategies_rule_id_fkey" FOREIGN KEY ("rule_id") REFERENCES "public"."rules"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
ALTER TABLE "rule_strategies" DROP COLUMN IF EXISTS "preset_id";