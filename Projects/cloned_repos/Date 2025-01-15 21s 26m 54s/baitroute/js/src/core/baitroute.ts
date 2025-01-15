import { promises as fs } from 'fs';
import { join } from 'path';
import { load } from 'js-yaml';
import type { EndpointConfig, BaitRouteOptions, Alert, AlertHandler } from './types';

export class BaitRoute {
    private readonly rulesDir: string;
    private readonly selectedRules?: string[];
    private endpoints: EndpointConfig[] = [];
    private alertHandler?: AlertHandler;

    constructor(options: BaitRouteOptions) {
        this.rulesDir = options.rulesDir;
        this.selectedRules = options.selectedRules;
    }

    async initialize(): Promise<void> {
        try {
            const files = await this.findYamlFiles(this.rulesDir);
            for (const file of files) {
                const ruleName = this.getRuleName(file);
                if (this.selectedRules && !this.selectedRules.includes(ruleName)) {
                    continue;
                }

                const content = await fs.readFile(file, 'utf8');
                const rules = load(content) as EndpointConfig[];
                if (Array.isArray(rules)) {
                    this.endpoints.push(...rules);
                }
            }
            
            const banner = ` ....
''. :   __
   \\|_.'  \`:       _.----._//_          Utku Sen's
  .'  .'.'-._   .'  _/ -._ \\)-.----O   ___       _ _   ___          _
 '._.'.'      '--''-'._   '--..--'    | _ ) __ _(_) |_| _ \\___ _  _| |_ ___
  .'.'___    /'---'. / ,-'            | _ \\/ _  | |  _|   / _ \\ || |  _/ -_)
_<__.-._))../ /'----'/.'_____:'.      |___/\\__,_|_|\\__|_|_\\___/\\_,_|\\__\\___|
:            \\ ]              :  '.     
:  Acme       \\\\              :    '.  A web honeypot library to create decoy
:              \\\\__           :    .'  endpoints to detect and mislead attackers
:_______________|__]__________:...'
`;
            process.stdout.write('\n');
            process.stdout.write(banner);
            process.stdout.write(`Successfully loaded ${this.endpoints.length} bait endpoints\n`);
            process.stdout.write('\n');
        } catch (error) {
            console.error('Error loading rules:', error);
            throw error;
        }
    }

    private async findYamlFiles(dir: string): Promise<string[]> {
        const files: string[] = [];
        const entries = await fs.readdir(dir, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = join(dir, entry.name);
            if (entry.isDirectory()) {
                files.push(...await this.findYamlFiles(fullPath));
            } else if (entry.isFile() && /\.ya?ml$/i.test(entry.name)) {
                files.push(fullPath);
            }
        }

        return files;
    }

    private getRuleName(filePath: string): string {
        const relativePath = filePath.slice(this.rulesDir.length + 1);
        return relativePath.replace(/\.ya?ml$/i, '');
    }

    setAlertHandler(handler: AlertHandler): void {
        this.alertHandler = handler;
    }

    protected async handleAlert(alert: Alert): Promise<void> {
        if (this.alertHandler) {
            await this.alertHandler(alert);
        }
    }

    getEndpoints(): EndpointConfig[] {
        return this.endpoints;
    }
} 