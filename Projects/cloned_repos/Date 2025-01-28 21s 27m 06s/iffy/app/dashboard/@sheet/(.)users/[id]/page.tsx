import { auth } from "@clerk/nextjs/server";
import { RecordUserDetail } from "@/app/dashboard/users/[id]/record-user";
import { redirect } from "next/navigation";
import { RouterSheet } from "@/components/router-sheet";

export default async function Page(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const { orgId } = await auth();
  if (!orgId) {
    redirect("/");
  }
  return (
    <RouterSheet title="User">
      <RecordUserDetail clerkOrganizationId={orgId} id={params.id} />
    </RouterSheet>
  );
}
