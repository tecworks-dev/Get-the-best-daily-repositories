import { clerkClient } from "@clerk/nextjs/server";

export async function formatClerkUser(clerkUserId: string) {
  try {
    const clerkUser = await (await clerkClient()).users.getUser(clerkUserId);
    if (clerkUser.firstName && clerkUser.lastName) {
      return `${clerkUser.firstName} ${clerkUser.lastName}`;
    }
    const email = clerkUser.emailAddresses.find((email) => email.id === clerkUser.primaryEmailAddressId)?.emailAddress;
    if (email) {
      return email;
    }
  } catch (error) {
    return clerkUserId;
  }
  return clerkUserId;
}
