import { app } from "/scripts/app.js";
import { ExtendedComfyWidgets, showVideoOutput } from "./extended_widgets.js";

app.registerExtension({
	name: "Comfy.VideoSave",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "SaveVideo [n-suite]") return;

		const onAdded = nodeType.prototype.onAdded;
		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onAdded = function () {
			onAdded?.apply(this, arguments);
			ExtendedComfyWidgets.VIDEO(this, "videoOutWidget", ["STRING"], "", app, "output");
		};
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			const paths = message?.text?.flat?.(Infinity) ?? message?.text ?? [];
			const fullPath = Array.isArray(paths) ? paths.at(-1) : paths;
			if (fullPath) showVideoOutput(fullPath, this);
		};
	},
});
