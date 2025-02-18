-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_CalendarEvent" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "feedId" TEXT NOT NULL,
    "googleEventId" TEXT,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "start" DATETIME NOT NULL,
    "end" DATETIME NOT NULL,
    "location" TEXT,
    "isRecurring" BOOLEAN NOT NULL DEFAULT false,
    "recurrenceRule" TEXT,
    "allDay" BOOLEAN NOT NULL DEFAULT false,
    "status" TEXT,
    "sequence" INTEGER,
    "created" DATETIME,
    "lastModified" DATETIME,
    "organizer" JSONB,
    "attendees" JSONB,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "isMaster" BOOLEAN NOT NULL DEFAULT false,
    "masterEventId" TEXT,
    "recurringEventId" TEXT,
    CONSTRAINT "CalendarEvent_feedId_fkey" FOREIGN KEY ("feedId") REFERENCES "CalendarFeed" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "CalendarEvent_masterEventId_fkey" FOREIGN KEY ("masterEventId") REFERENCES "CalendarEvent" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_CalendarEvent" ("allDay", "attendees", "created", "createdAt", "description", "end", "feedId", "googleEventId", "id", "isRecurring", "lastModified", "location", "organizer", "recurrenceRule", "sequence", "start", "status", "title", "updatedAt") SELECT "allDay", "attendees", "created", "createdAt", "description", "end", "feedId", "googleEventId", "id", "isRecurring", "lastModified", "location", "organizer", "recurrenceRule", "sequence", "start", "status", "title", "updatedAt" FROM "CalendarEvent";
DROP TABLE "CalendarEvent";
ALTER TABLE "new_CalendarEvent" RENAME TO "CalendarEvent";
CREATE INDEX "CalendarEvent_feedId_idx" ON "CalendarEvent"("feedId");
CREATE INDEX "CalendarEvent_start_end_idx" ON "CalendarEvent"("start", "end");
CREATE INDEX "CalendarEvent_googleEventId_idx" ON "CalendarEvent"("googleEventId");
CREATE INDEX "CalendarEvent_masterEventId_idx" ON "CalendarEvent"("masterEventId");
CREATE INDEX "CalendarEvent_recurringEventId_idx" ON "CalendarEvent"("recurringEventId");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
