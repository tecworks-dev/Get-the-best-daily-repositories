import { router } from "../trpc";

import { userRouter } from "./user";
import { recordRouter } from "./record";
import { appealRouter } from "./appeal";

export const appRouter = router({
  user: userRouter,
  record: recordRouter,
  appeal: appealRouter,
});

export type AppRouter = typeof appRouter;
