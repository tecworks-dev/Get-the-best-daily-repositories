import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const fontOptions = [
  { value: "inter", label: "Inter" },
  { value: "roboto", label: "Roboto" },
  { value: "lato", label: "Lato" },
  { value: "merriweather", label: "Merriweather" },
  { value: "fira-code", label: "Fira Code" },
];
export function FontSelector({
  selectedFont,
  setSelectedFont,
}: {
  selectedFont: string;
  setSelectedFont: (font: string) => void;
}) {
  return (
    <Select value={selectedFont} onValueChange={setSelectedFont}>
      <SelectTrigger className="w-[180px]">
        <SelectValue placeholder="Select a font" />
      </SelectTrigger>
      <SelectContent>
        {fontOptions.map((font) => (
          <SelectItem key={font.value} value={font.value}>
            {font.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
