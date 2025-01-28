import * as schema from "@/db/schema";

type RecordBase = typeof schema.records.$inferSelect;
type ModerationToRule = typeof schema.moderationsToRules.$inferSelect & {
  rule: typeof schema.rules.$inferSelect & {
    preset: typeof schema.presets.$inferSelect | null;
  };
};
type Moderation = typeof schema.moderations.$inferSelect & {
  moderationsToRules: ModerationToRule[];
};

type Record = RecordBase & {
  moderations: Moderation[];
};

export type { Record };
