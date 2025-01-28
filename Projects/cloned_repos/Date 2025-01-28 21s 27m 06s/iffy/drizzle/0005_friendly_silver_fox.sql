DROP MATERIALIZED VIEW IF EXISTS "public"."moderations_analytics_daily";
CREATE MATERIALIZED VIEW "public"."moderations_analytics_daily" AS (
  SELECT
    date_trunc('day', "moderations"."created_at" AT TIME ZONE 'UTC') AS time,
    "moderations"."clerk_organization_id" AS clerk_organization_id,
    COUNT(*)::int AS moderations,
    COUNT(*) FILTER (WHERE "moderations"."status" = 'Flagged')::int AS flagged
  FROM "moderations"
  WHERE "moderations"."created_at" AT TIME ZONE 'UTC' >= date_trunc('day', now() AT TIME ZONE 'UTC') - INTERVAL '29 days'
  GROUP BY time, "moderations"."clerk_organization_id"
);
CREATE UNIQUE INDEX IF NOT EXISTS "moderations_analytics_daily_time_clerk_organization_id_idx" ON "public"."moderations_analytics_daily" ("time", "clerk_organization_id");

DROP MATERIALIZED VIEW IF EXISTS "public"."moderations_analytics_hourly";
CREATE MATERIALIZED VIEW "public"."moderations_analytics_hourly" AS (
  SELECT
    date_trunc('hour', "moderations"."created_at" AT TIME ZONE 'UTC') AS time,
    "moderations"."clerk_organization_id" AS clerk_organization_id,
    COUNT(*)::int AS moderations,
    COUNT(*) FILTER (WHERE "moderations"."status" = 'Flagged')::int AS flagged
  FROM "moderations"
  WHERE "moderations"."created_at" AT TIME ZONE 'UTC' >= date_trunc('hour', now() AT TIME ZONE 'UTC') - INTERVAL '23 hours'
  GROUP BY time, "moderations"."clerk_organization_id"
);
CREATE UNIQUE INDEX IF NOT EXISTS "moderations_analytics_hourly_time_clerk_organization_id_idx" ON "public"."moderations_analytics_hourly" ("time", "clerk_organization_id");
