import type {ActionFunction} from '@remix-run/node';
import getPipeWithPipesAsTools from '~/../baseai/pipes/pipe-with-pipes-as-tools';
import {Pipe} from '@baseai/core';

export const action: ActionFunction = async ({request}) => {
	const runOptions = await request.json();

	// 1. Initiate the Pipe.
	// const pipe = new Pipe(getPipeTinyLlama());
	const pipe = new Pipe(getPipeWithPipesAsTools());

	// 2. Run the pipe
	const result = await pipe.run(runOptions);

	// 3. Return the response stringified.
	return new Response(JSON.stringify(result));
};