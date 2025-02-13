/*
  Warnings:

  - The values [lue,azel] on the enum `EyeColorEnum` will be removed. If these variants are still used in the database, this will fail.

*/
-- AlterEnum
BEGIN;
CREATE TYPE "EyeColorEnum_new" AS ENUM ('Brown', 'Blue', 'Hazel', 'Gray');
ALTER TABLE "Model" ALTER COLUMN "eyeColor" TYPE "EyeColorEnum_new" USING ("eyeColor"::text::"EyeColorEnum_new");
ALTER TYPE "EyeColorEnum" RENAME TO "EyeColorEnum_old";
ALTER TYPE "EyeColorEnum_new" RENAME TO "EyeColorEnum";
DROP TYPE "EyeColorEnum_old";
COMMIT;
