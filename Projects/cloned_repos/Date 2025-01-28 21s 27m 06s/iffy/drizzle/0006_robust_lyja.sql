ALTER TABLE "examples" RENAME COLUMN "file_urls" TO "image_urls";--> statement-breakpoint
ALTER TABLE "records" RENAME COLUMN "file_urls" TO "image_urls";--> statement-breakpoint
ALTER TABLE "records" ADD COLUMN "external_urls" text[] DEFAULT '{}' NOT NULL;--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "moderations_org_status_idx" ON "moderations" USING btree ("clerk_organization_id" text_ops,"status");