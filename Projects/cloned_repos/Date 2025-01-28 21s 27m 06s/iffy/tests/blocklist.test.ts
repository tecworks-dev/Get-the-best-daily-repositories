import { expect, test } from "vitest";
import { checkBlocklist } from "@/strategies/blocklist";

test("blocklist should correctly block provided words", async () => {
  expect(await checkBlocklist("Hello, world!", ["hello"])).toEqual([true, ["hello"]]);
});

test("blocklist should not block words that are not in the list", async () => {
  expect(await checkBlocklist("Hello, world!", [])).toEqual([false, null]);
});
