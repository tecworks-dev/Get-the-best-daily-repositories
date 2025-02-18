-- CreateTable
CREATE TABLE "AutoScheduleSettings" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "workDays" TEXT NOT NULL DEFAULT '[]',
    "workHourStart" INTEGER NOT NULL,
    "workHourEnd" INTEGER NOT NULL,
    "selectedCalendars" TEXT NOT NULL DEFAULT '[]',
    "bufferMinutes" INTEGER NOT NULL DEFAULT 15,
    "highEnergyStart" INTEGER,
    "highEnergyEnd" INTEGER,
    "mediumEnergyStart" INTEGER,
    "mediumEnergyEnd" INTEGER,
    "lowEnergyStart" INTEGER,
    "lowEnergyEnd" INTEGER,
    "groupByProject" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "AutoScheduleSettings_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_Task" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "status" TEXT NOT NULL,
    "dueDate" DATETIME,
    "duration" INTEGER,
    "energyLevel" TEXT,
    "preferredTime" TEXT,
    "isAutoScheduled" BOOLEAN NOT NULL DEFAULT false,
    "scheduledStart" DATETIME,
    "scheduledEnd" DATETIME,
    "scheduleScore" REAL,
    "lastScheduled" DATETIME,
    "scheduleLocked" BOOLEAN NOT NULL DEFAULT false,
    "isRecurring" BOOLEAN NOT NULL DEFAULT false,
    "recurrenceRule" TEXT,
    "lastCompletedDate" DATETIME,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "projectId" TEXT,
    CONSTRAINT "Task_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "Project" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);
INSERT INTO "new_Task" ("createdAt", "description", "dueDate", "duration", "energyLevel", "id", "isRecurring", "lastCompletedDate", "preferredTime", "projectId", "recurrenceRule", "status", "title", "updatedAt") SELECT "createdAt", "description", "dueDate", "duration", "energyLevel", "id", "isRecurring", "lastCompletedDate", "preferredTime", "projectId", "recurrenceRule", "status", "title", "updatedAt" FROM "Task";
DROP TABLE "Task";
ALTER TABLE "new_Task" RENAME TO "Task";
CREATE INDEX "Task_status_idx" ON "Task"("status");
CREATE INDEX "Task_dueDate_idx" ON "Task"("dueDate");
CREATE INDEX "Task_projectId_idx" ON "Task"("projectId");
CREATE INDEX "Task_isRecurring_idx" ON "Task"("isRecurring");
CREATE INDEX "Task_isAutoScheduled_idx" ON "Task"("isAutoScheduled");
CREATE INDEX "Task_scheduledStart_scheduledEnd_idx" ON "Task"("scheduledStart", "scheduledEnd");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;

-- CreateIndex
CREATE UNIQUE INDEX "AutoScheduleSettings_userId_key" ON "AutoScheduleSettings"("userId");

-- CreateIndex
CREATE INDEX "AutoScheduleSettings_userId_idx" ON "AutoScheduleSettings"("userId");
