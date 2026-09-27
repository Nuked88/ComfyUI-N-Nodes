import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
	name: "n.DynamicPrompt",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "DynamicPrompt [n-suite]") return;

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function () {
			onExecuted?.apply(this, arguments);
			const widgets = this.widgets ?? [];
			const cached = widgets.find((widget) => widget.name === "cached");
			for (const widget of widgets.filter((item) => item.name === "text")) {
				this.removeWidget(widget);
			}
			if (cached?.value === "NO") {
				const result = ComfyWidgets.STRING(
					this,
					"text",
					["STRING", { multiline: true }],
					app,
				);
				result.widget.value = Math.floor(Math.random() * 10000);
			}
		};
	},
});
