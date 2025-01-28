import { Template } from "../types";

const defaultContent = {
  subject: "Your account has been permanently banned",
  heading: "Your account has been permanently banned from {{ ORGANIZATION_NAME }}",
  body: "Hey,\n\nDue to repeated or severe violations of {{ ORGANIZATION_NAME }}'s rules, your account has been permanently banned.\n\nAs a result, all sales of your products have been stopped and your account has been permanently disabled.\n\nThis decision is final and cannot be appealed.\n\nThank you for your understanding.",
};

export default {
  defaultContent,
} satisfies Template;
