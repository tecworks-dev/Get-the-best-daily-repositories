-- Current sql file was generated after introspecting the database
-- If you want to run this migration please uncomment this code before executing migrations

DO $$ BEGIN
    CREATE TYPE "public"."AppealActionStatus" AS ENUM('Open', 'Rejected', 'Approved');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "public"."EmailTemplateType" AS ENUM('Suspended', 'Compliant', 'Banned');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "public"."MessageStatus" AS ENUM('Pending', 'Delivered');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "public"."MessageType" AS ENUM('Outbound', 'Inbound');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "public"."ModerationStatus" AS ENUM('Compliant', 'Flagged');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "public"."RecordUserActionStatus" AS ENUM('Compliant', 'Suspended', 'Banned');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "public"."Via" AS ENUM('Inbound', 'Manual', 'Automation', 'AI');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "public"."WebhookEventStatus" AS ENUM('Pending', 'Sent', 'Failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS "moderations" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"status" "ModerationStatus" NOT NULL,
	"pending" boolean DEFAULT false NOT NULL,
	"via" "Via" DEFAULT 'AI' NOT NULL,
	"reasoning" text,
	"record_id" text NOT NULL,
	"ruleset_id" text,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL,
	"clerk_user_id" text,
	"test_mode" boolean DEFAULT false NOT NULL,
	"tokens" integer DEFAULT 0 NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "rulesets" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"name" text NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "record_user_actions" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"record_user_id" text NOT NULL,
	"status" "RecordUserActionStatus" NOT NULL,
	"via" "Via" DEFAULT 'Automation' NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"clerk_user_id" text
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "ruleset_categories" (
	"id" text PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"description" text,
	"prompt" text,
	"blocklist" text[] DEFAULT '{"RAY"}',
	"ruleset_id" text NOT NULL,
	"enabled" boolean DEFAULT true NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "examples" (
	"id" text PRIMARY KEY NOT NULL,
	"text" text NOT NULL,
	"file_urls" text[],
	"comment" text,
	"category_id" text NOT NULL,
	"flagged" boolean NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "messages" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"record_user_action_id" text NOT NULL,
	"to_id" text,
	"from_id" text,
	"type" "MessageType" NOT NULL,
	"subject" text,
	"text" text NOT NULL,
	"status" "MessageStatus" NOT NULL,
	"appeal_id" text,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"sort" serial NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "record_users" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"client_id" text NOT NULL,
	"client_url" text,
	"stripe_account_id" text,
	"email" text,
	"name" text,
	"username" text,
	"protected" boolean DEFAULT false NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL,
	"sort" serial NOT NULL,
	"action_status" "RecordUserActionStatus",
	"action_status_created_at" timestamp(3),
	"flagged_records_count" integer DEFAULT 0 NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "webhook_endpoints" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"url" text NOT NULL,
	"secret" text NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "appeals" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"record_user_action_id" text NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL,
	"sort" serial NOT NULL,
	"action_status" "AppealActionStatus",
	"action_status_created_at" timestamp(3)
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "appeal_actions" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"appeal_id" text NOT NULL,
	"status" "AppealActionStatus" NOT NULL,
	"via" "Via" DEFAULT 'Inbound' NOT NULL,
	"clerk_user_id" text,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "api_keys" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"clerk_user_id" text NOT NULL,
	"name" text NOT NULL,
	"encrypted_key" text NOT NULL,
	"encrypted_key_hash" text,
	"last_used_at" timestamp(3),
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL,
	"deleted_at" timestamp(3)
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "organization_settings" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"emails_enabled" boolean DEFAULT false NOT NULL,
	"test_mode_enabled" boolean DEFAULT true NOT NULL,
	"appeals_enabled" boolean DEFAULT false NOT NULL,
	"categories_enabled" boolean DEFAULT false NOT NULL,
	"blocklist" text[] DEFAULT '{"RAY"}',
	"stripe_api_key" text,
	"moderation_percentage" double precision DEFAULT 100 NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "openai_moderation_parameters" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"sexual" double precision NOT NULL,
	"sexual_minors" double precision NOT NULL,
	"harassment" double precision NOT NULL,
	"harassment_threatening" double precision NOT NULL,
	"hate" double precision NOT NULL,
	"hate_threatening" double precision NOT NULL,
	"illicit" double precision NOT NULL,
	"illicit_violent" double precision NOT NULL,
	"self_harm" double precision NOT NULL,
	"self_harm_intent" double precision NOT NULL,
	"self_harm_instructions" double precision NOT NULL,
	"violence" double precision NOT NULL,
	"violence_graphic" double precision NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "_ModerationToRulesetCategory" (
	"A" text NOT NULL,
	"B" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "custom_email_configurations" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"name" text NOT NULL,
	"address" text NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "records" (
	"id" text PRIMARY KEY NOT NULL,
	"clerk_organization_id" text NOT NULL,
	"client_id" text NOT NULL,
	"client_url" text,
	"name" text NOT NULL,
	"entity" text NOT NULL,
	"text" text NOT NULL,
	"file_urls" text[],
	"record_user_id" text,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL,
	"sort" serial NOT NULL,
	"moderation_status" "ModerationStatus",
	"moderation_status_created_at" timestamp(3),
	"moderation_pending" boolean DEFAULT false NOT NULL,
	"moderation_pending_created_at" timestamp(3),
	"deleted_at" timestamp(3)
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "webhook_events" (
	"id" text PRIMARY KEY NOT NULL,
	"webhook_endpoint_id" text NOT NULL,
	"event_type" text NOT NULL,
	"payload" jsonb NOT NULL,
	"status" "WebhookEventStatus" NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "category_openai_moderation_parameters" (
	"id" text PRIMARY KEY NOT NULL,
	"category_id" text NOT NULL,
	"sexual" double precision NOT NULL,
	"sexual_minors" double precision NOT NULL,
	"harassment" double precision NOT NULL,
	"harassment_threatening" double precision NOT NULL,
	"hate" double precision NOT NULL,
	"hate_threatening" double precision NOT NULL,
	"illicit" double precision NOT NULL,
	"illicit_violent" double precision NOT NULL,
	"self_harm" double precision NOT NULL,
	"self_harm_intent" double precision NOT NULL,
	"self_harm_instructions" double precision NOT NULL,
	"violence" double precision NOT NULL,
	"violence_graphic" double precision NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "email_templates" (
	"clerk_organization_id" text NOT NULL,
	"type" "EmailTemplateType" NOT NULL,
	"content" jsonb NOT NULL,
	"created_at" timestamp(3) DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp(3) NOT NULL,
	CONSTRAINT "email_templates_pkey" PRIMARY KEY("clerk_organization_id","type")
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "moderations" ADD CONSTRAINT "moderations_record_id_fkey" FOREIGN KEY ("record_id") REFERENCES "public"."records"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "moderations" ADD CONSTRAINT "moderations_ruleset_id_fkey" FOREIGN KEY ("ruleset_id") REFERENCES "public"."rulesets"("id") ON DELETE set null ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "record_user_actions" ADD CONSTRAINT "record_user_actions_record_user_id_fkey" FOREIGN KEY ("record_user_id") REFERENCES "public"."record_users"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "ruleset_categories" ADD CONSTRAINT "ruleset_categories_ruleset_id_fkey" FOREIGN KEY ("ruleset_id") REFERENCES "public"."rulesets"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "examples" ADD CONSTRAINT "examples_category_id_fkey" FOREIGN KEY ("category_id") REFERENCES "public"."ruleset_categories"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "messages" ADD CONSTRAINT "messages_appeal_id_fkey" FOREIGN KEY ("appeal_id") REFERENCES "public"."appeals"("id") ON DELETE set null ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "messages" ADD CONSTRAINT "messages_from_id_fkey" FOREIGN KEY ("from_id") REFERENCES "public"."record_users"("id") ON DELETE set null ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "messages" ADD CONSTRAINT "messages_record_user_action_id_fkey" FOREIGN KEY ("record_user_action_id") REFERENCES "public"."record_user_actions"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "messages" ADD CONSTRAINT "messages_to_id_fkey" FOREIGN KEY ("to_id") REFERENCES "public"."record_users"("id") ON DELETE set null ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "appeals" ADD CONSTRAINT "appeals_record_user_action_id_fkey" FOREIGN KEY ("record_user_action_id") REFERENCES "public"."record_user_actions"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "appeal_actions" ADD CONSTRAINT "appeal_actions_appeal_id_fkey" FOREIGN KEY ("appeal_id") REFERENCES "public"."appeals"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "_ModerationToRulesetCategory" ADD CONSTRAINT "_ModerationToRulesetCategory_A_fkey" FOREIGN KEY ("A") REFERENCES "public"."moderations"("id") ON DELETE cascade ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "_ModerationToRulesetCategory" ADD CONSTRAINT "_ModerationToRulesetCategory_B_fkey" FOREIGN KEY ("B") REFERENCES "public"."ruleset_categories"("id") ON DELETE cascade ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "records" ADD CONSTRAINT "records_record_user_id_fkey" FOREIGN KEY ("record_user_id") REFERENCES "public"."record_users"("id") ON DELETE set null ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "webhook_events" ADD CONSTRAINT "webhook_events_webhook_endpoint_id_fkey" FOREIGN KEY ("webhook_endpoint_id") REFERENCES "public"."webhook_endpoints"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "category_openai_moderation_parameters" ADD CONSTRAINT "category_openai_moderation_parameters_category_id_fkey" FOREIGN KEY ("category_id") REFERENCES "public"."ruleset_categories"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "moderations_record_id_idx" ON "moderations" USING btree ("record_id" text_ops);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "record_user_actions_record_user_id_idx" ON "record_user_actions" USING btree ("record_user_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "messages_sort_key" ON "messages" USING btree ("sort" int4_ops);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "record_users_clerk_organization_id_idx" ON "record_users" USING btree ("clerk_organization_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "record_users_client_id_key" ON "record_users" USING btree ("client_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "record_users_sort_key" ON "record_users" USING btree ("sort" int4_ops);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "appeals_clerk_organization_id_idx" ON "appeals" USING btree ("clerk_organization_id" text_ops);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "appeals_record_user_action_id_idx" ON "appeals" USING btree ("record_user_action_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "appeals_record_user_action_id_key" ON "appeals" USING btree ("record_user_action_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "appeals_sort_key" ON "appeals" USING btree ("sort" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "api_keys_encrypted_key_hash_key" ON "api_keys" USING btree ("encrypted_key_hash" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "api_keys_encrypted_key_key" ON "api_keys" USING btree ("encrypted_key" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "organization_settings_clerk_organization_id_key" ON "organization_settings" USING btree ("clerk_organization_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "openai_moderation_parameters_clerk_organization_id_key" ON "openai_moderation_parameters" USING btree ("clerk_organization_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "_ModerationToRulesetCategory_AB_unique" ON "_ModerationToRulesetCategory" USING btree ("A" text_ops,"B" text_ops);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "_ModerationToRulesetCategory_B_index" ON "_ModerationToRulesetCategory" USING btree ("B" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "custom_email_configurations_clerk_organization_id_key" ON "custom_email_configurations" USING btree ("clerk_organization_id" text_ops);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "records_clerk_organization_id_idx" ON "records" USING btree ("clerk_organization_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "records_client_id_key" ON "records" USING btree ("client_id" text_ops);--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "records_record_user_id_idx" ON "records" USING btree ("record_user_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "records_sort_key" ON "records" USING btree ("sort" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "category_openai_moderation_parameters_category_id_key" ON "category_openai_moderation_parameters" USING btree ("category_id" text_ops);
