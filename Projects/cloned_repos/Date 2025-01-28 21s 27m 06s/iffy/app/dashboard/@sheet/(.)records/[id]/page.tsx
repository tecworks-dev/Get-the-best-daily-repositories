import { redirect } from "next/navigation";
import { RecordDetail } from "@/app/dashboard/records/[id]/record";
import { auth } from "@clerk/nextjs/server";
import { RouterSheet } from "@/components/router-sheet";

export default async function Page(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const { orgId } = await auth();
  if (!orgId) {
    redirect("/");
  }
  return (
    <RouterSheet title="Record">
      <RecordDetail clerkOrganizationId={orgId} id={params.id} />
    </RouterSheet>
  );
}
