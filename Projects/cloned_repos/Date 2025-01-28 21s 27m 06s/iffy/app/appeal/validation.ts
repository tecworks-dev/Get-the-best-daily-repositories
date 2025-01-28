import { z } from "zod";

export const submitAppealSchema = z.object({
  text: z.string().min(1, "Appeal text is required").max(1000, "Appeal text must be less than 1000 characters"),
});
