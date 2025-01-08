import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { useUser } from "@/context/useUser";
import LLMSettings from "@/components/SettingsModal/SettingsComponents/LLMSettings";
export default function SettingsAlert() {
  const { alertForUser, setAlertForUser } = useUser();

  return (
    <Dialog open={alertForUser} onOpenChange={setAlertForUser}>
      <DialogContent>
        <DialogTitle>LLM Settings</DialogTitle>
        <DialogDescription>
          Please add an API key or Select Local Model Deployment
          <br />
          *Local Model Deployment requires Ollama to be installed and running
        </DialogDescription>
        <LLMSettings />
      </DialogContent>
    </Dialog>
  );
}
