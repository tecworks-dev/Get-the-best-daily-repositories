import {runAppleScript} from 'run-applescript';

async function sendMessage(phoneNumber: string, message: string) {
    const escapedMessage = message.replace(/"/g, '\\"');
    const result = await runAppleScript(`
tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy "${phoneNumber}"
    send "${escapedMessage}" to targetBuddy
end tell`);
    return result;
}

export default { sendMessage };
