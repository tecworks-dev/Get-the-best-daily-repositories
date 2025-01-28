import { TRPCError } from "@trpc/server";
import { middleware, procedure } from "../trpc";

const protectedProcedure = procedure.use(
  middleware(async ({ ctx, next }) => {
    const { clerkOrganizationId, clerkUserId } = ctx;
    if (!clerkOrganizationId || !clerkUserId) {
      throw new TRPCError({ code: "UNAUTHORIZED" });
    }

    return next({
      ctx: {
        clerkOrganizationId,
        clerkUserId,
      },
    });
  }),
);

export default protectedProcedure;
