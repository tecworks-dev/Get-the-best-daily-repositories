import { inngest } from "../client";
import db from "@/db";
import * as schema from "@/db/schema";
export const refreshAnalyticsViews = inngest.createFunction(
  { id: "refresh-analytics-views" },
  { cron: "*/5 * * * *" },
  async ({ step }) => {
    await step.run("refresh-moderations-analytics-daily", async () => {
      await db.refreshMaterializedView(schema.moderationsAnalyticsDaily).concurrently();
    });
    await step.run("refresh-moderations-analytics-hourly", async () => {
      await db.refreshMaterializedView(schema.moderationsAnalyticsHourly).concurrently();
    });
  },
);

export default [refreshAnalyticsViews];
