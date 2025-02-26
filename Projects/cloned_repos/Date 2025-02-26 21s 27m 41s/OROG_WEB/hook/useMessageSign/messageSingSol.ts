const messageSingSol = async (
  providerSol: any,
  message: string = "Hello OROG"
) => {
  if (providerSol) {
    try {
      const encodedMessage = new TextEncoder().encode(message);
      const data = await providerSol.signMessage(encodedMessage, "utf8");
      return data;
    } catch (e) {
      console.log("messageSingSol", e);
      throw new Error(`messageSingSol error  ${e}`);
    }
  }
  return "";
};

export default messageSingSol;
