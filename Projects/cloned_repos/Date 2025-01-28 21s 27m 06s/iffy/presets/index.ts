import { close } from "@iffy/app/db";
import { updatePresets } from "./presets";

async function main() {
  await updatePresets();
}

main()
  .then(() => {
    console.log("Updating presets completed successfully.");
    close();
  })
  .catch((e) => {
    console.error(e);
    close();
    process.exit(1);
  });
