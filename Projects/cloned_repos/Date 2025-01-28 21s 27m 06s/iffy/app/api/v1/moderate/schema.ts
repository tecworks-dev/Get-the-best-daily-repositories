import { z } from "zod";

export const ModerateRequestData = z
  .object({
    clientId: z.string(),
    clientUrl: z.string().url().optional(),
    name: z.string(),
    entity: z.string(),
    content: z.union([
      z.string(),
      z.object({
        text: z.string(),
        imageUrls: z.array(z.string().url()).optional(),
        externalUrls: z.array(z.string().url()).optional(),
      }),
    ]),
    user: z
      .object({
        clientId: z.string(),
        clientUrl: z.string().url().optional(),
        stripeAccountId: z.string().optional(),
        email: z.string().optional(),
        name: z.string().optional(),
        username: z.string().optional(),
        protected: z.boolean().optional(),
      })
      .optional(),
  })
  .strict();

export type ModerateRequestData = z.infer<typeof ModerateRequestData>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export const moderateAdapter = (data: unknown) => {
  if (!isRecord(data)) {
    return data;
  }
  const { text, fileUrls, ...rest } = data;
  return { content: { text, imageUrls: fileUrls }, ...rest };
};
