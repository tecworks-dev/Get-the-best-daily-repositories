import { auth } from "@clerk/nextjs/server";
import DataTable from "./data-table";
import { redirect } from "next/navigation";

const Users = async () => {
  const { orgId } = await auth();
  if (!orgId) {
    redirect("/");
  }

  return <DataTable clerkOrganizationId={orgId} />;
};

export default Users;
