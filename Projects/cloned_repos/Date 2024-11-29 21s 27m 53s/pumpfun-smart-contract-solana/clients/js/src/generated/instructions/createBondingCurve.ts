/**
 * This code was AUTOGENERATED using the kinobi library.
 * Please DO NOT EDIT THIS FILE, instead use visitors
 * to add features, then rerun kinobi to update it.
 *
 * @see https://github.com/metaplex-foundation/kinobi
 */

import { Context, Option, OptionOrNullable, Pda, PublicKey, Signer, TransactionBuilder, publicKey, transactionBuilder } from '@metaplex-foundation/umi';
import { Serializer, array, i64, mapSerializer, option, string, struct, u8 } from '@metaplex-foundation/umi/serializers';
import { ResolvedAccount, ResolvedAccountsWithIndices, getAccountMetasAndSigners } from '../shared';

// Accounts.
export type CreateBondingCurveInstructionAccounts = {
    mint: Signer;
    creator: Signer;
    bondingCurve: PublicKey | Pda;
    bondingCurveTokenAccount: PublicKey | Pda;
    global: PublicKey | Pda;
    metadata: PublicKey | Pda;
    systemProgram?: PublicKey | Pda;
    tokenProgram?: PublicKey | Pda;
    associatedTokenProgram: PublicKey | Pda;
    tokenMetadataProgram?: PublicKey | Pda;
    rent?: PublicKey | Pda;
    clock: PublicKey | Pda;
    eventAuthority: PublicKey | Pda;
    program: PublicKey | Pda;
};

  // Data.
  export type CreateBondingCurveInstructionData = { discriminator: Array<number>; name: string; symbol: string; uri: string; startTime: Option<bigint>;  };

export type CreateBondingCurveInstructionDataArgs = { name: string; symbol: string; uri: string; startTime: OptionOrNullable<number | bigint>;  };


  export function getCreateBondingCurveInstructionDataSerializer(): Serializer<CreateBondingCurveInstructionDataArgs, CreateBondingCurveInstructionData> {
  return mapSerializer<CreateBondingCurveInstructionDataArgs, any, CreateBondingCurveInstructionData>(struct<CreateBondingCurveInstructionData>([['discriminator', array(u8(), { size: 8 })], ['name', string()], ['symbol', string()], ['uri', string()], ['startTime', option(i64())]], { description: 'CreateBondingCurveInstructionData' }), (value) => ({ ...value, discriminator: [94, 139, 158, 50, 69, 95, 8, 45] }) ) as Serializer<CreateBondingCurveInstructionDataArgs, CreateBondingCurveInstructionData>;
}



  
  // Args.
      export type CreateBondingCurveInstructionArgs =           CreateBondingCurveInstructionDataArgs
      ;
  
// Instruction.
export function createBondingCurve(
  context: Pick<Context, "programs">,
                        input: CreateBondingCurveInstructionAccounts & CreateBondingCurveInstructionArgs,
      ): TransactionBuilder {
  // Program ID.
  const programId = context.programs.getPublicKey('pumpFun', 'DkgjYaaXrunwvqWT3JmJb29BMbmet7mWUifQeMQLSEQH');

  // Accounts.
  const resolvedAccounts = {
          mint: { index: 0, isWritable: true as boolean, value: input.mint ?? null },
          creator: { index: 1, isWritable: true as boolean, value: input.creator ?? null },
          bondingCurve: { index: 2, isWritable: true as boolean, value: input.bondingCurve ?? null },
          bondingCurveTokenAccount: { index: 3, isWritable: true as boolean, value: input.bondingCurveTokenAccount ?? null },
          global: { index: 4, isWritable: false as boolean, value: input.global ?? null },
          metadata: { index: 5, isWritable: true as boolean, value: input.metadata ?? null },
          systemProgram: { index: 6, isWritable: false as boolean, value: input.systemProgram ?? null },
          tokenProgram: { index: 7, isWritable: false as boolean, value: input.tokenProgram ?? null },
          associatedTokenProgram: { index: 8, isWritable: false as boolean, value: input.associatedTokenProgram ?? null },
          tokenMetadataProgram: { index: 9, isWritable: false as boolean, value: input.tokenMetadataProgram ?? null },
          rent: { index: 10, isWritable: false as boolean, value: input.rent ?? null },
          clock: { index: 11, isWritable: false as boolean, value: input.clock ?? null },
          eventAuthority: { index: 12, isWritable: false as boolean, value: input.eventAuthority ?? null },
          program: { index: 13, isWritable: false as boolean, value: input.program ?? null },
      } satisfies ResolvedAccountsWithIndices;

      // Arguments.
    const resolvedArgs: CreateBondingCurveInstructionArgs = { ...input };
  
    // Default values.
  if (!resolvedAccounts.systemProgram.value) {
        resolvedAccounts.systemProgram.value = context.programs.getPublicKey('splSystem', '11111111111111111111111111111111');
resolvedAccounts.systemProgram.isWritable = false
      }
      if (!resolvedAccounts.tokenProgram.value) {
        resolvedAccounts.tokenProgram.value = context.programs.getPublicKey('splToken', 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA');
resolvedAccounts.tokenProgram.isWritable = false
      }
      if (!resolvedAccounts.tokenMetadataProgram.value) {
        resolvedAccounts.tokenMetadataProgram.value = context.programs.getPublicKey('mplTokenMetadata', 'metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s');
resolvedAccounts.tokenMetadataProgram.isWritable = false
      }
      if (!resolvedAccounts.rent.value) {
        resolvedAccounts.rent.value = publicKey('SysvarRent111111111111111111111111111111111');
      }
      
  // Accounts in order.
      const orderedAccounts: ResolvedAccount[] = Object.values(resolvedAccounts).sort((a,b) => a.index - b.index);
  
  
  // Keys and Signers.
  const [keys, signers] = getAccountMetasAndSigners(orderedAccounts, "programId", programId);

  // Data.
      const data = getCreateBondingCurveInstructionDataSerializer().serialize(resolvedArgs as CreateBondingCurveInstructionDataArgs);
  
  // Bytes Created On Chain.
      const bytesCreatedOnChain = 0;
  
  return transactionBuilder([{ instruction: { keys, programId, data }, signers, bytesCreatedOnChain }]);
}
