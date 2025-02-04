import { auth, signIn } from '@/app/(auth)/auth';
import { getChatsByUserId } from '@/lib/db/queries';

export async function GET() {
  let session = await auth();

  // If no session exists, create an anonymous session
  if (!session?.user) {
    try {
      const result = await signIn('credentials', {
        redirect: false,
      });

      if (result?.error) {
        console.error('Failed to create anonymous session:', result.error);
        return Response.json('Failed to create anonymous session', {
          status: 500,
        });
      }

      session = await auth();

      if (!session?.user) {
        console.error('Failed to get session after creation');
        return Response.json('Failed to create session', { status: 500 });
      }
    } catch (error) {
      console.error('Error creating anonymous session:', error);
      return Response.json('Failed to create anonymous session', {
        status: 500,
      });
    }
  }

  // biome-ignore lint: Forbidden non-null assertion.
  const chats = await getChatsByUserId({ id: session.user.id! });
  return Response.json(chats);
}
