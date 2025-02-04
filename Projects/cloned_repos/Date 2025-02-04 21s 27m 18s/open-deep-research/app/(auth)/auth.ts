import { compare } from 'bcrypt-ts';
import NextAuth, { type User, type Session } from 'next-auth';
import Credentials from 'next-auth/providers/credentials';

import { getUser, createUser } from '@/lib/db/queries';
import { authConfig } from './auth.config';

interface ExtendedSession extends Session {
  user: User;
}

async function createAnonymousUser() {
  const anonymousEmail = `anon_${Date.now()}@anonymous.user`;
  const anonymousPassword = `anon_${Date.now()}_${Math.random().toString(36).slice(2)}`;

  try {
    await createUser(anonymousEmail, anonymousPassword);
    const [user] = await getUser(anonymousEmail);
    return user;
  } catch (error) {
    console.error('Failed to create anonymous user:', error);
    return null;
  }
}

export const {
  handlers: { GET, POST },
  auth,
  signIn,
  signOut,
} = NextAuth({
  ...authConfig,
  providers: [
    Credentials({
      credentials: {},
      async authorize({ email, password }: any) {
        // Handle anonymous access
        if (!email && !password) {
          return await createAnonymousUser();
        }

        // Handle regular authentication
        const users = await getUser(email);
        if (users.length === 0) return null;
        // biome-ignore lint: Forbidden non-null assertion.
        const passwordsMatch = await compare(password, users[0].password!);
        if (!passwordsMatch) return null;
        return users[0] as any;
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      } else if (!token.id) {
        // Create anonymous user if no user exists
        const anonymousUser = await createAnonymousUser();
        if (anonymousUser) {
          token.id = anonymousUser.id;
          token.email = anonymousUser.email;
        }
      }

      return token;
    },
    async session({
      session,
      token,
    }: {
      session: ExtendedSession;
      token: any;
    }) {
      if (session.user) {
        session.user.id = token.id as string;
        session.user.email = token.email as string;
      }

      return session;
    },
  },
});
