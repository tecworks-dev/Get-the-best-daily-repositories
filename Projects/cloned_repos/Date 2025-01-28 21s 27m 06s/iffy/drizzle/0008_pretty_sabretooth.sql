UPDATE "records" SET "image_urls" = '{}' WHERE "image_urls" IS NULL;--> statement-breakpoint
ALTER TABLE "records" ALTER COLUMN "image_urls" SET DEFAULT '{}';--> statement-breakpoint
ALTER TABLE "records" ALTER COLUMN "image_urls" SET NOT NULL;--> statement-breakpoint
