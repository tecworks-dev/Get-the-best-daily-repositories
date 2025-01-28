import { z } from "zod";

export type DefaultTemplateContent = {
  subject: string;
  heading: string;
  body: string;
};

export type Template = {
  defaultContent: DefaultTemplateContent;
};

export type RenderedTemplate = {
  html: string;
  subject: string;
  heading: string;
  body: string;
};
