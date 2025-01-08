import LocalModels from "./LocalModels";
import ExternalApi from "./ExternalApi";
import { useSysSettings } from "@/context/useSysSettings";
import SourceSelect from "./SourceSelect";

export default function LLMSettings() {
  const { sourceType } = useSysSettings();

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold mb-4">API Keys & Models</h2>
        <div className="space-y-6">
          <SourceSelect />

          {sourceType === "external" && <ExternalApi />}

          {sourceType === "local" && <LocalModels />}
        </div>
      </div>
    </div>
  );
}
