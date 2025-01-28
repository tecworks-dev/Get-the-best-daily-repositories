"use client";

import { useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { updateEmailTemplate } from "./actions";
import { zodResolver } from "@hookform/resolvers/zod";
import { updateEmailTemplateSchema } from "./schema";
import * as schema from "@/db/schema";
import { DefaultTemplateContent } from "@/emails/types";

type EmailTemplateType = (typeof schema.emailTemplateType.enumValues)[number];

export default function ContentForm({ type, content }: { type: EmailTemplateType; content: DefaultTemplateContent }) {
  const updateEmailTemplateWithTemplate = updateEmailTemplate.bind(null, type);

  const form = useForm({
    resolver: zodResolver(updateEmailTemplateSchema),
    defaultValues: content,
  });

  const onSubmit = form.handleSubmit(async (values) => {
    const result = await updateEmailTemplateWithTemplate(values);
    if (result?.serverError) {
      form.setError("body", { message: "Content contains invalid template syntax" });
    }
  });

  return (
    <div className="relative flex flex-col items-start gap-8">
      <form onSubmit={onSubmit} className="grid w-full items-start gap-6">
        <fieldset className="grid gap-6 rounded-md border p-4 dark:border-zinc-700">
          <legend className="-ml-1 px-1 text-sm font-medium">Content</legend>
          <div className="grid gap-3">
            <Label htmlFor="subject">Subject</Label>
            <Input id="subject" {...form.register("subject")} />
            {form.formState.errors.subject && (
              <p className="text-sm text-red-500">{form.formState.errors.subject.message}</p>
            )}
          </div>
          <div className="grid gap-3">
            <Label htmlFor="heading">Heading</Label>
            <Textarea id="heading" className="min-h-24" {...form.register("heading")} />
            {form.formState.errors.heading && (
              <p className="text-sm text-red-500">{form.formState.errors.heading.message}</p>
            )}
          </div>
          <div className="grid gap-3">
            <Label htmlFor="body">Body</Label>
            <Textarea id="body" className="min-h-96" {...form.register("body")} />
            {form.formState.errors.body && <p className="text-sm text-red-500">{form.formState.errors.body.message}</p>}
          </div>
        </fieldset>
        <Button type="submit" disabled={form.formState.isSubmitting}>
          {form.formState.isSubmitting ? "Saving..." : "Save"}
        </Button>
      </form>
    </div>
  );
}
