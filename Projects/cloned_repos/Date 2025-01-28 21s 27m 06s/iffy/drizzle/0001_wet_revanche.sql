DO $$ BEGIN
 ALTER TABLE "messages" ADD CONSTRAINT "messages_sort_unique" UNIQUE("sort");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "record_users" ADD CONSTRAINT "record_users_client_id_unique" UNIQUE("client_id");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "record_users" ADD CONSTRAINT "record_users_sort_unique" UNIQUE("sort");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "appeals" ADD CONSTRAINT "appeals_record_user_action_id_unique" UNIQUE("record_user_action_id");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "appeals" ADD CONSTRAINT "appeals_sort_unique" UNIQUE("sort");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "api_keys" ADD CONSTRAINT "api_keys_encrypted_key_unique" UNIQUE("encrypted_key");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "api_keys" ADD CONSTRAINT "api_keys_encrypted_key_hash_unique" UNIQUE("encrypted_key_hash");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "organization_settings" ADD CONSTRAINT "organization_settings_clerk_organization_id_unique" UNIQUE("clerk_organization_id");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "openai_moderation_parameters" ADD CONSTRAINT "openai_moderation_parameters_clerk_organization_id_unique" UNIQUE("clerk_organization_id");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "custom_email_configurations" ADD CONSTRAINT "custom_email_configurations_clerk_organization_id_unique" UNIQUE("clerk_organization_id");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "records" ADD CONSTRAINT "records_client_id_unique" UNIQUE("client_id");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "records" ADD CONSTRAINT "records_sort_unique" UNIQUE("sort");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "category_openai_moderation_parameters" ADD CONSTRAINT "category_openai_moderation_parameters_category_id_unique" UNIQUE("category_id");
EXCEPTION
 WHEN duplicate_table THEN null;
END $$;
