import { Template } from "../types";

const defaultContent = {
  subject: "Your account has been suspended",
  heading: "Your account has been suspended for violating {{ ORGANIZATION_NAME }}'s rules",
  body: "Hey,\n\nIt looks like one or more of your records is in violation of {{ ORGANIZATION_NAME }}'s rules.\n\nAs a result, we've temporarily paused sales of the affected products and payouts for your account.\n\nYour account will remain suspended until you update or remove the offending content.\n",
};

export default {
  defaultContent,
} satisfies Template;
