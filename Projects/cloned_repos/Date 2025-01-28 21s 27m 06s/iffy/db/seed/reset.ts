import { reset } from "drizzle-seed";
import db, { close } from "@/db";
import * as schema from "@/db/schema";

async function main() {
  await reset(db, schema);
}

main()
  .then(() => {
    console.log("Reset successfully.");
    close();
  })
  .catch((e) => {
    console.error(e);
    close();
    process.exit(1);
  });
