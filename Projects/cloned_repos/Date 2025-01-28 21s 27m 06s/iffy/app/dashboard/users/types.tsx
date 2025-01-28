import * as schema from "@/db/schema";

type RecordUser = typeof schema.recordUsers.$inferSelect & {
  actions: (typeof schema.recordUserActions.$inferSelect)[];
};

type RecordUserDetail = typeof schema.recordUsers.$inferSelect & {
  actions: (typeof schema.recordUserActions.$inferSelect & {
    appeal: typeof schema.appeals.$inferSelect | null;
  })[];
};

export type { RecordUser, RecordUserDetail };
