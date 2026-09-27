import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "n.ModelDownloadNotice",
    setup() {
        api.addEventListener("n-suite-model-download", (event) => {
            const { model, message } = event.detail;
            app.extensionManager.toast.add({
                severity: "info",
                summary: `${model} download started`,
                detail: message,
                life: 15000,
            });
        });
    },
});
