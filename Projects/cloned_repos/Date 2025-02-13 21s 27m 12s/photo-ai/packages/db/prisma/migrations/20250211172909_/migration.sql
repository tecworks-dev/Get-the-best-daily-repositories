/*
  Warnings:

  - The values [AsianAmerican,EastAsian,SouthEastAsian,SouthAsian,MiddleEastern] on the enum `EthenecityEnum` will be removed. If these variants are still used in the database, this will fail.

*/
-- AlterEnum
BEGIN;
CREATE TYPE "EthenecityEnum_new" AS ENUM ('White', 'Black', 'Asian American', 'East Asian', 'South East Asian', 'South Asian', 'Middle Eastern', 'Pacific', 'Hispanic');
ALTER TABLE "Model" ALTER COLUMN "ethinicity" TYPE "EthenecityEnum_new" USING ("ethinicity"::text::"EthenecityEnum_new");
ALTER TYPE "EthenecityEnum" RENAME TO "EthenecityEnum_old";
ALTER TYPE "EthenecityEnum_new" RENAME TO "EthenecityEnum";
DROP TYPE "EthenecityEnum_old";
COMMIT;
