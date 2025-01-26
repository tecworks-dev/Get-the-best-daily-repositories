<script>
	import { onMount } from 'svelte';
	import { Toaster, toast } from 'svelte-sonner';
	import { fly } from 'svelte/transition';

	let originalTextBox,
		anonymizedText,
		llmResponse,
		deanonymizedText,
		gettingResponse = false,
		entities = {};

	$: deanonymizedText = llmResponse?.replace(/<PII_(\w+)_(\d+)>/g, (match, entity, index) => {
		return entities[entity]?.[parseInt(index) - 1] || match;
	});

	onMount(() => {
		originalTextBox.focus();
	});
</script>

<svelte:head>
	<title>Elara</title>
	<meta name="description" content="A web-based anonymizer/de-anonymizer for LLMs." />
</svelte:head>

<Toaster position="top-center" />

<form
	on:submit|preventDefault={async (e) => {
		toast.promise(
			() =>
				new Promise(async (resolve) => {
					if (!originalTextBox.value.trim()) {
						toast.warning('Please enter some text to anonymize.');
						return;
					}

					gettingResponse = true;

					const response = await fetch('http://localhost:8000/', {
						method: 'POST',
						headers: {
							'Content-Type': 'application/json'
						},
						body: JSON.stringify({
							text: originalTextBox.value
						})
					});

					const jsonResponse = await response.json();

					anonymizedText = jsonResponse.anonymized_text;
					entities = jsonResponse.entities;
					gettingResponse = false;

					resolve();
				}),
			{
				loading:
					'Fetching response from Python webserver (please make sure that the Python server is running!)...',
				success: 'Text anonymized successfully!',
				error: 'Failed to anonymize text. Please try again.'
			}
		);
	}}
>
	<!-- svelte-ignore a11y_consider_explicit_label -->
	<div class="mx-auto grid max-w-7xl gap-4 p-4 md:grid-cols-2">
		<div class="flex flex-col items-center justify-center gap-6">
			<div class="relative h-[400px] w-full">
				<textarea
					class="h-full w-full resize-none rounded-lg border border-gray-300 p-4 pt-8 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
					placeholder="Type some text here..."
					name="input"
					bind:this={originalTextBox}
					on:keypress={(e) => {
						if (e.key === 'Enter' && !e.shiftKey) {
							e.preventDefault();
							e.target.form.requestSubmit();
						}
					}}
					disabled={gettingResponse}
				></textarea>
				<div
					class="absolute left-0 top-0 select-none rounded-br-lg rounded-tl-lg bg-blue-500 px-2 py-1 text-sm font-bold leading-tight text-white"
				>
					ORIGINAL TEXT
				</div>
				<button
					type="submit"
					class="absolute right-0 top-0 select-none rounded-bl-lg rounded-tr-lg bg-gray-700 px-2 py-1 text-sm font-bold leading-tight text-white transition-colors hover:bg-gray-800"
					disabled={gettingResponse}
				>
					SUBMIT
				</button>
			</div>
		</div>
		<div
			class={'relative h-[400px] whitespace-pre-wrap rounded-lg border border-gray-300 bg-white p-4 shadow-lg ' +
				(anonymizedText ? 'overflow-y-auto pt-10' : '')}
		>
			{#if anonymizedText}
				{anonymizedText}
			{:else}
				<div
					class="mx-auto flex h-full w-full select-none items-center justify-center text-center leading-tight text-gray-500"
				>
					Please write and submit some<br />text for anonymization.
				</div>
			{/if}
			<div
				class="absolute left-0 top-0 select-none rounded-br-lg rounded-tl-lg bg-rose-500 px-2 py-1 text-sm font-bold leading-tight text-white"
			>
				ANONYMIZED TEXT
			</div>
			<button
				class="absolute right-[8px] top-[8px] flex h-6 w-6 items-center justify-center rounded-lg bg-gray-700 p-1 transition-colors hover:bg-gray-800"
				type="button"
				on:click={() => {
					navigator.clipboard.writeText(anonymizedText);
					toast.success('Anonymized response copied to clipboard!');
				}}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					width="24"
					height="24"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
					class="icon icon-tabler icons-tabler-outline icon-tabler-copy w-4 text-white"
					><path stroke="none" d="M0 0h24v24H0z" fill="none" /><path
						d="M7 7m0 2.667a2.667 2.667 0 0 1 2.667 -2.667h8.666a2.667 2.667 0 0 1 2.667 2.667v8.666a2.667 2.667 0 0 1 -2.667 2.667h-8.666a2.667 2.667 0 0 1 -2.667 -2.667z"
					/><path
						d="M4.012 16.737a2.005 2.005 0 0 1 -1.012 -1.737v-10c0 -1.1 .9 -2 2 -2h10c.75 0 1.158 .385 1.5 1"
					/></svg
				>
			</button>
		</div>

		{#if anonymizedText}
			<div
				class="flex flex-col items-center justify-center gap-6"
				in:fly={{ duration: 300, y: 20 }}
			>
				<div class="relative h-[400px] w-full">
					<textarea
						class="h-full w-full resize-none rounded-lg border border-gray-300 p-4 pt-8 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
						placeholder="Type some text here..."
						name="input"
						bind:value={llmResponse}
					></textarea>
					<div
						class="absolute left-0 top-0 select-none rounded-br-lg rounded-tl-lg bg-emerald-500 px-2 py-1 text-sm font-bold leading-tight text-white"
					>
						ANONYMIZED LLM RESPONSE
					</div>
				</div>
			</div>
			<div
				class={'relative h-[400px] whitespace-pre-wrap rounded-lg border border-gray-300 bg-white p-4 shadow-lg ' +
					(deanonymizedText ? 'overflow-y-auto pt-10' : '')}
				in:fly={{ duration: 300, y: 20 }}
			>
				{#if deanonymizedText}
					{deanonymizedText}
				{:else}
					<div
						class="mx-auto flex h-full w-full select-none items-center justify-center text-center leading-tight text-gray-500"
					>
						Please paste an LLM response<br />for de-anonymization.
					</div>
				{/if}
				<div
					class="absolute left-0 top-0 select-none rounded-br-lg rounded-tl-lg bg-purple-500 px-2 py-1 text-sm font-bold leading-tight text-white"
				>
					DE-ANONYMIZED LLM RESPONSE
				</div>
				<button
					class="absolute right-[8px] top-[8px] flex h-6 w-6 items-center justify-center rounded-lg bg-gray-700 p-1 transition-colors hover:bg-gray-800"
					type="button"
					on:click={() => {
						navigator.clipboard.writeText(deanonymizedText);
						toast.success('De-anonymized LLM response copied to clipboard!');
					}}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="24"
						height="24"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
						class="icon icon-tabler icons-tabler-outline icon-tabler-copy w-4 text-white"
						><path stroke="none" d="M0 0h24v24H0z" fill="none" /><path
							d="M7 7m0 2.667a2.667 2.667 0 0 1 2.667 -2.667h8.666a2.667 2.667 0 0 1 2.667 2.667v8.666a2.667 2.667 0 0 1 -2.667 2.667h-8.666a2.667 2.667 0 0 1 -2.667 -2.667z"
						/><path
							d="M4.012 16.737a2.005 2.005 0 0 1 -1.012 -1.737v-10c0 -1.1 .9 -2 2 -2h10c.75 0 1.158 .385 1.5 1"
						/></svg
					>
				</button>
			</div>
		{/if}
	</div>
</form>
