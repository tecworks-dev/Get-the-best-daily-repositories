function initializeSettingsPage() {
    document.getElementById("save-settings-button").addEventListener("click", (event) => {
        event.preventDefault();
        let formData = {};
        document.querySelectorAll("#settings-form input").forEach(input => {
            if (input.type === "number") {
                formData[input.id] = parseFloat(input.value);
            } else {
                formData[input.id] = input.value;
            }
        });
        socket.emit("save_settings", formData);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    initializeSettingsPage();
});

