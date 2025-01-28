import { Template } from "../types";

const defaultContent = {
  subject: "Your account has been unsuspended",
  heading: "Your account has been unsuspended",
  body: "Hey,\n\nCongratulations, your account has been unsuspended!\n\nWe've re-enabled sales of your products and payouts for your account.\n\nWe hope you'll continue to follow {{ ORGANIZATION_NAME }}'s rules and provide a great experience for your customers.",
};

export default {
  defaultContent,
} satisfies Template;
