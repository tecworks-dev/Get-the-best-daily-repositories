CREATE TABLE IF NOT EXISTS "public"."veindexer" (
        "id" BIGINT GENERATED ALWAYS AS IDENTITY,
        "chain_info_id" INT NOT NULL,
        "height" BIGINT NOT NULL,
        "validator_hex_address_id" INT NOT NULL,
        "status" SMALLINT NOT NULL, 
        "vote_extension_length" INT NOT NULL,
        "timestamp" timestamptz NOT NULL,
        PRIMARY KEY ("id", "chain_info_id"),
        CONSTRAINT fk_chain_info_id FOREIGN KEY (chain_info_id) REFERENCES meta.chain_info (id) ON DELETE CASCADE ON UPDATE CASCADE,
        CONSTRAINT fk_validator_hex_address_id FOREIGN KEY (validator_hex_address_id, chain_info_id) REFERENCES meta.validator_info (id, chain_info_id),
        CONSTRAINT uniq_validator_hex_address_by_height UNIQUE ("chain_info_id","height","validator_hex_address_id")
    )
PARTITION BY
    LIST ("chain_info_id");

CREATE INDEX IF NOT EXISTS veindexer_idx_01 ON public.veindexer (height);
CREATE INDEX IF NOT EXISTS  veindexer_idx_02 ON public.veindexer (validator_hex_address_id, height);
CREATE INDEX IF NOT EXISTS veindexer_idx_03 ON public.veindexer USING btree (chain_info_id, validator_hex_address_id, height asc);