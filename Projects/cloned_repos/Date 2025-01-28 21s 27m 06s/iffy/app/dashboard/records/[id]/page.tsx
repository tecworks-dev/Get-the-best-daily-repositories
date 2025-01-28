import { redirect } from "next/navigation";
import { RecordDetail } from "./record";
import { auth } from "@clerk/nextjs/server";

export default async function Page(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const { orgId } = await auth();
  if (!orgId) {
    redirect("/");
  }
  return <RecordDetail clerkOrganizationId={orgId} id={params.id} />;
}
