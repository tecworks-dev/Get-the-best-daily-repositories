import { inngest as client } from "@/inngest/client";
import functions from "@/inngest/functions";
import { serve } from "inngest/next";

export const maxDuration = 300;

export const { GET, POST, PUT } = serve({
  client,
  functions,
});
