import { auth } from "@clerk/nextjs/server";
import { notFound, redirect } from "next/navigation";
import { Appeals } from "./appeals";
import { findOrCreateOrganizationSettings } from "@/services/organization-settings";

const InboxLayout = async ({ children }: { children: React.ReactNode }) => {
  const { orgId } = await auth();
  if (!orgId) {
    redirect("/");
  }

  const organizationSettings = await findOrCreateOrganizationSettings(orgId);
  if (!organizationSettings.appealsEnabled) {
    return notFound();
  }

  return <Appeals clerkOrganizationId={orgId}>{children}</Appeals>;
};

export default InboxLayout;
