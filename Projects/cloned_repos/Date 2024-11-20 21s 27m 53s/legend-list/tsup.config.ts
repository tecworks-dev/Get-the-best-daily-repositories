import path from 'node:path';
import fs from 'node:fs';
import { defineConfig } from 'tsup';
// @ts-expect-error It says import assertions don't work, but they do
import pkg from './package.json' assert { type: 'json' };

const Exclude = new Set(['.DS_Store']);

const external = [
    '@babel/types',
    'next',
    'next/router',
    'react',
    'react-native',
    'react-native-mmkv',
    '@react-native-async-storage/async-storage',
    '@tanstack/react-query',
    '@tanstack/query-core',
    '@legendapp/state',
    '@legendapp/state/config',
    '@legendapp/state/persist',
    '@legendapp/state/sync',
    '@legendapp/state/sync-plugins/crud',
    '@legendapp/state/sync-plugins/tanstack-query',
    '@legendapp/state/react',
    '@legendapp/state/helpers/fetch',
    '@legendapp/state/react-reactive/enableReactive',
    'firebase/auth',
    'firebase/database',
];

export default defineConfig({
    entry: ['src/index.ts'],
    format: ['cjs', 'esm'],
    external,
    dts: true,
    treeshake: true,
    splitting: false,
    clean: true,
});
