import * as schema from "./tables";
import { pgMaterializedView, timestamp, text, integer } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export const moderationsAnalyticsHourly = pgMaterializedView("moderations_analytics_hourly", {
  time: timestamp("time").notNull(),
  clerkOrganizationId: text("clerk_organization_id").notNull(),
  moderations: integer("moderations").notNull(),
  flagged: integer("flagged").notNull(),
}).as(sql`
  SELECT
    date_trunc('hour', ${schema.moderations.createdAt} AT TIME ZONE 'UTC') AS time,
    ${schema.moderations.clerkOrganizationId} AS clerk_organization_id,
    COUNT(*)::int AS moderations,
    COUNT(*) FILTER (WHERE ${schema.moderations.status} = 'Flagged')::int AS flagged
  FROM ${schema.moderations}
  WHERE ${schema.moderations.createdAt} AT TIME ZONE 'UTC' >= date_trunc('hour', now() AT TIME ZONE 'UTC') - INTERVAL '23 hours'
  GROUP BY time, ${schema.moderations.clerkOrganizationId}
`);

export const moderationsAnalyticsDaily = pgMaterializedView("moderations_analytics_daily", {
  time: timestamp("time").notNull(),
  clerkOrganizationId: text("clerk_organization_id").notNull(),
  moderations: integer("moderations").notNull(),
  flagged: integer("flagged").notNull(),
}).as(sql`
  SELECT
    date_trunc('day', ${schema.moderations.createdAt} AT TIME ZONE 'UTC') AS time,
    ${schema.moderations.clerkOrganizationId} AS clerk_organization_id,
    COUNT(*)::int AS moderations,
    COUNT(*) FILTER (WHERE ${schema.moderations.status} = 'Flagged')::int AS flagged
  FROM ${schema.moderations}
  WHERE ${schema.moderations.createdAt} AT TIME ZONE 'UTC' >= date_trunc('day', now() AT TIME ZONE 'UTC') - INTERVAL '29 days'
  GROUP BY time, ${schema.moderations.clerkOrganizationId}
`);
