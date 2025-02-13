"use client";

import { useState } from "react";
import { SelectModel } from "./Models";
import { PackCard, TPack } from "./PackCard";

export function PacksClient({packs}: {
    packs: TPack[]
}) {
    const [selectedModelId, setSelectedModelId] = useState<string>();

    return <div className="justify-center flex">
        <div className="pt-4 max-w-2xl"> 
            <div>
                <SelectModel selectedModel={selectedModelId} setSelectedModel={setSelectedModelId} />
                <div className="text-2xl max-w-4xl pt-4">
                    Select Pack
                </div>
                <div className="grid md:grid-cols-3 gap-4 p-4 grids-cols-1">
                    {packs.map(p => <PackCard selectedModelId={selectedModelId!} {...p} />)}
                </div>
            </div>
        </div>
    </div>
}