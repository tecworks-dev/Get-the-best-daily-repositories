<template>
  <main class="flex h-[calc(100vh-180px)] items-center">
    <div class="mx-auto w-full overflow-hidden rounded-lg bg-white shadow-xl sm:max-w-sm">
      <div class="relative p-5 text-center">
        <div class="my-5">
          <div class="mx-auto mb-4 flex justify-center text-red-500">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="h-16 w-16">
              <path
                fill-rule="evenodd"
                d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zm-1.72 6.97a.75.75 0 10-1.06 1.06L10.94 12l-1.72 1.72a.75.75 0 101.06 1.06L12 13.06l1.72 1.72a.75.75 0 101.06-1.06L13.06 12l1.72-1.72a.75.75 0 10-1.06-1.06L12 10.94l-1.72-1.72z"
                clip-rule="evenodd" />
            </svg>
          </div>
          <div>
            <h3 class="text-2xl font-medium text-secondary-900">404</h3>
            <div class="mt-4 px-2 text-sm text-secondary-500">
              <h1>Page not found</h1>
            </div>
          </div>
          <div class="mt-5 flex justify-end gap-3">
            <NuxtLink
              to="/"
              class="flex-1 rounded-lg border border-primary-500 bg-primary-500 px-4 py-2 text-center text-sm font-medium text-white shadow-sm transition-all hover:border-primary-700 hover:bg-primary-700 focus:ring focus:ring-primary-200 disabled:cursor-not-allowed disabled:border-primary-300 disabled:bg-primary-300">
              Back to Home
            </NuxtLink>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<script>
export default {
  props: ["error"],
  layout: "error", // you can set a custom layout for the error page
};
</script>
