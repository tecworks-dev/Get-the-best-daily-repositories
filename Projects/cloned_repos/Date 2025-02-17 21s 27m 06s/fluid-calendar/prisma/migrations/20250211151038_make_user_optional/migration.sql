-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_CalendarFeed" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "url" TEXT,
    "type" TEXT NOT NULL,
    "color" TEXT,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "lastSync" DATETIME,
    "error" TEXT,
    "channelId" TEXT,
    "resourceId" TEXT,
    "channelExpiration" DATETIME,
    "userId" TEXT,
    CONSTRAINT "CalendarFeed_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "new_CalendarFeed" ("channelExpiration", "channelId", "color", "createdAt", "enabled", "error", "id", "lastSync", "name", "resourceId", "type", "updatedAt", "url", "userId") SELECT "channelExpiration", "channelId", "color", "createdAt", "enabled", "error", "id", "lastSync", "name", "resourceId", "type", "updatedAt", "url", "userId" FROM "CalendarFeed";
DROP TABLE "CalendarFeed";
ALTER TABLE "new_CalendarFeed" RENAME TO "CalendarFeed";
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
