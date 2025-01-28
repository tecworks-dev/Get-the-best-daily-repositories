DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = '_ModerationToRulesetCategory_A_B_pk'
    ) THEN
        ALTER TABLE "_ModerationToRulesetCategory"
        ADD CONSTRAINT "_ModerationToRulesetCategory_A_B_pk"
        PRIMARY KEY("A","B");
    END IF;
END $$;
