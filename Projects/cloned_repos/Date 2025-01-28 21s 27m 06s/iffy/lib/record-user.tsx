import * as schema from "@/db/schema";

type RecordUser = typeof schema.recordUsers.$inferSelect;

export function formatRecordUser(user: RecordUser) {
  let primary = user.clientId;
  if (user.email) primary = user.email;
  if (user.username) primary = user.username;
  if (user.name) primary = user.name;
  return primary;
}

export function getRecordUserSecondaryParts(user: RecordUser) {
  let secondary = ["No email provided"];
  if (user.email) secondary = [user.clientId];
  if (user.username && user.email) secondary = [user.email];
  if (user.name && user.username && user.email) secondary = [user.username, user.email];
  return secondary;
}

export function formatRecordUserSecondary(user: RecordUser) {
  return getRecordUserSecondaryParts(user)[0];
}

export function formatRecordUserCompact(user: RecordUser) {
  let primary = user.clientId;
  // in compact situations the ranking of properties is different than above
  if (user.name) primary = user.name;
  if (user.email) primary = user.email;
  if (user.username) primary = user.username;
  return primary;
}

export function formatRecordUserCompactSecondary(user: RecordUser) {
  let secondary = "No email provided";
  // in compact situations the ranking of properties is different than above
  if (user.name) secondary = user.clientId;
  if (user.email && user.name) secondary = user.name;
  if (user.username && user.email) secondary = user.email;
  return secondary;
}
