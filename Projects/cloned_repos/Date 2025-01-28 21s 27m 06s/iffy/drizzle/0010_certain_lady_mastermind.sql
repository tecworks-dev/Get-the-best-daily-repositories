CREATE TYPE "public"."StrategyType" AS ENUM('Blocklist', 'OpenAI', 'Prompt');--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "presets" (
	"id" text PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"description" text,
	"created_at" timestamp (3) DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "rules" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"name" text,
	"description" text,
	"preset_id" text,
	"ruleset_id" text NOT NULL,
	"created_at" timestamp (3) DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "strategies" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text,
	"type" "StrategyType" NOT NULL,
	"rule_id" text,
	"preset_id" text,
	"options" jsonb NOT NULL,
	"created_at" timestamp (3) DEFAULT now() NOT NULL,
	"updated_at" timestamp (3) DEFAULT now() NOT NULL
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "rules" ADD CONSTRAINT "rules_ruleset_id_fkey" FOREIGN KEY ("ruleset_id") REFERENCES "public"."rulesets"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "strategies" ADD CONSTRAINT "strategies_rule_id_fkey" FOREIGN KEY ("rule_id") REFERENCES "public"."rules"("id") ON DELETE set null ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "strategies" ADD CONSTRAINT "strategies_preset_id_fkey" FOREIGN KEY ("preset_id") REFERENCES "public"."presets"("id") ON DELETE set null ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
