import { auth } from "@clerk/nextjs/server";
import { RecordUserDetail } from "./record-user";
import { redirect } from "next/navigation";

export default async function Page(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const { orgId } = await auth();
  if (!orgId) {
    redirect("/");
  }
  return <RecordUserDetail clerkOrganizationId={orgId} id={params.id} />;
}
