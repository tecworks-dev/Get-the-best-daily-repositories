import { auth, signIn } from '@/app/(auth)/auth';
import { getSuggestionsByDocumentId } from '@/lib/db/queries';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const documentId = searchParams.get('documentId');

  if (!documentId) {
    return new Response('Not Found', { status: 404 });
  }

  let session = await auth();

  // If no session exists, create an anonymous session
  if (!session?.user) {
    try {
      const result = await signIn('credentials', {
        redirect: false,
      });

      if (result?.error) {
        console.error('Failed to create anonymous session:', result.error);
        return new Response('Failed to create anonymous session', {
          status: 500,
        });
      }

      session = await auth();

      if (!session?.user) {
        console.error('Failed to get session after creation');
        return new Response('Failed to create session', { status: 500 });
      }
    } catch (error) {
      console.error('Error creating anonymous session:', error);
      return new Response('Failed to create anonymous session', {
        status: 500,
      });
    }
  }

  const suggestions = await getSuggestionsByDocumentId({
    documentId,
  });

  const [suggestion] = suggestions;

  if (!suggestion) {
    return Response.json([], { status: 200 });
  }

  if (suggestion.userId !== session.user.id) {
    return new Response('Unauthorized', { status: 401 });
  }

  return Response.json(suggestions, { status: 200 });
}
