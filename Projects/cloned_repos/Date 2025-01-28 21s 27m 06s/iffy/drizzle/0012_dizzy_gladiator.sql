ALTER TABLE "strategies" DROP CONSTRAINT "strategies_rule_id_fkey";
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "strategies" ADD CONSTRAINT "strategies_rule_id_fkey" FOREIGN KEY ("rule_id") REFERENCES "public"."rules"("id") ON DELETE restrict ON UPDATE cascade;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
