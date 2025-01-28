CREATE TABLE IF NOT EXISTS "moderations_to_rules" (
	"moderation_id" text NOT NULL,
	"rule_id" text NOT NULL,
	CONSTRAINT "moderations_to_rules_moderation_id_rule_id_pk" PRIMARY KEY("moderation_id","rule_id")
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "moderations_to_rules" ADD CONSTRAINT "moderations_to_rules_moderation_id_fkey" FOREIGN KEY ("moderation_id") REFERENCES "public"."moderations"("id") ON DELETE cascade ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "moderations_to_rules" ADD CONSTRAINT "moderations_to_rules_rule_id_fkey" FOREIGN KEY ("rule_id") REFERENCES "public"."rules"("id") ON DELETE cascade ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
