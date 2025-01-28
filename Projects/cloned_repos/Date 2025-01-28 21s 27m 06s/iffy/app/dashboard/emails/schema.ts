import { z } from "zod";

export const updateEmailTemplateSchema = z.object({
  subject: z.string(),
  heading: z.string(),
  body: z.string(),
});
