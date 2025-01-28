import { findOrCreateOrganizationSettings } from "@/services/organization-settings";
import { Settings } from "./settings";
import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";

export default async function SettingsPage() {
  const { orgId } = await auth();

  if (!orgId) {
    redirect("/");
  }

  const organizationSettings = await findOrCreateOrganizationSettings(orgId);

  return (
    <div className="px-12 py-8">
      <Settings organizationSettings={organizationSettings} />
    </div>
  );
}
