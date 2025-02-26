import { Connection } from '@solana/web3.js';

const checkTransactionStatus = async (
  connection: Connection,
  signature: string
) => {
  try {
    const confirmation = await connection.confirmTransaction(
      signature,
      "finalized"
    );

    if (confirmation.value.err) {
      return false;
    } else {
      return true;
    }
  } catch (e) {
    console.log(e);
    return false;
  }
};

export default checkTransactionStatus;
