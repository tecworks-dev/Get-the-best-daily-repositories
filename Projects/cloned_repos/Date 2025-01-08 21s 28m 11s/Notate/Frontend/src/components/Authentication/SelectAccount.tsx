import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";
import { useView } from "@/context/useView";
import { Button } from "@/components/ui/button";
import { useUser } from "@/context/useUser";
import { useSysSettings } from "@/context/useSysSettings";

export default function SelectAccount({ users }: { users: User[] }) {
  const { setActiveView } = useView();
  const { activeUser, setActiveUser } = useUser();
  const { setSettings } = useSysSettings();

  const fetchSettings = async () => {
    if (activeUser) {
      const userSettings = await window.electron.getUserSettings(activeUser.id);
      setSettings(userSettings);
    }
  };
  
  const handleSelectAccount = (user: User) => {
    setActiveUser(user);
    fetchSettings();
    setActiveView("Chat");
  };

  return (
    <div className="h-screen flex flex-col p-4 bg-gradient-to-b from-background via-muted/50 to-muted">
      <div className="max-w-md w-full mx-auto flex flex-col h-full mt-2">
        {activeUser ? (
          <div className="flex-none">
            <h2 className="text-3xl font-bold mb-6 text-center bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/60">
              Selected Account: {activeUser.name}
            </h2>
          </div>
        ) : (
          <h2 className="text-3xl font-bold mb-6 text-center flex-none bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/60">
            Select an Account
          </h2>
        )}
        <div className="flex flex-col flex-1 overflow-hidden">
          <div className="flex-1 overflow-y-auto space-y-4 p-4 min-h-0">
            {users.map((user) => (
              <Card
                key={user.name}
                className="hover:shadow-lg transition-all duration-300 cursor-pointer border border-border/50 bg-card/90 backdrop-blur-sm hover:scale-[1.02] shadow-sm"
                onClick={() => handleSelectAccount(user)}
              >
                <CardContent className="flex items-center p-4">
                  <Avatar className="h-14 w-14 mr-4 ring-2 ring-primary">
                    <AvatarFallback className="bg-primary/20 text-primary font-semibold">
                      {user.name.charAt(0)}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <h3 className="font-semibold text-lg text-foreground">
                      {user.name}
                    </h3>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          <div className="mt-2 mb-2">
            <Button
              onClick={() => setActiveView("Signup")}
              className="w-full h-12 text-lg shadow-lg hover:shadow-xl transition-shadow"
            >
              Add New Account
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
