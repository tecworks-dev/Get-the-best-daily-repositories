import { faker } from "@faker-js/faker";
import sample from "lodash/sample";
import db from "../index";
import * as schema from "../schema";

const COUNT = 50;

export async function seedRecordUsers(clerkOrganizationId: string) {
  const recordUsers = await db
    .insert(schema.recordUsers)
    .values(
      [...Array(COUNT)].map(() => {
        const firstName = faker.person.firstName();
        const lastName = faker.person.lastName();
        return {
          clerkOrganizationId,
          clientId: `user_${faker.string.nanoid(10)}`,
          email: faker.internet.email({ firstName, lastName }).toLocaleLowerCase(),
          name: faker.person.fullName({ firstName, lastName }),
          username: faker.internet.userName({ firstName, lastName }).toLocaleLowerCase(),
          createdAt: faker.date.recent({ days: 10 }),
          protected: sample([true, false]),
        };
      }),
    )
    .returning();
  console.log("Seeded Record Users");

  return recordUsers;
}
