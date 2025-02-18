import { prisma } from "@/lib/prisma";

export async function getGoogleCredentials() {
  try {
    const settings = await prisma.systemSettings.findFirst();
    if (settings) {
      return {
        clientId: settings.googleClientId || process.env.GOOGLE_CLIENT_ID!,
        clientSecret:
          settings.googleClientSecret || process.env.GOOGLE_CLIENT_SECRET!,
      };
    }
  } catch (error) {
    console.error("Failed to get system settings:", error);
  }

  // Fallback to environment variables
  return {
    clientId: process.env.GOOGLE_CLIENT_ID!,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
  };
}
