import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ExtendedComfyWidgets, showVideoInput } from "./extended_widgets.js";

const VIDEO_TYPES = new Set(["video/mp4", "video/webm", "image/gif"]);

async function uploadFile(file, updateNode, node, pasted = false) {
	const videoWidget = node.widgets?.find((widget) => widget.name === "video");
	if (!videoWidget) return false;

	try {
		const body = new FormData();
		body.append("image", file);
		body.append("subfolder", pasted ? "pasted" : "n-suite");
		const response = await api.fetchApi("/upload/image", { method: "POST", body });
		if (!response.ok) {
			alert(`${response.status} - ${response.statusText}`);
			return false;
		}

		const data = await response.json();
		const value = data.name;
		const previewPath = data.subfolder ? `${data.subfolder}/${value}` : value;
		if (!videoWidget.options.values.includes(value)) videoWidget.options.values.push(value);
		if (updateNode) {
			const oldValue = videoWidget.value;
			videoWidget.value = value;
			videoWidget.callback?.(value);
			node.onWidgetChanged?.(videoWidget.name, value, oldValue, videoWidget);
			showVideoInput(previewPath, node);
		}
		return true;
	} catch (error) {
		console.error("N-Suite video upload failed", error);
		alert(String(error));
		return false;
	}
}

app.registerExtension({
	name: "Comfy.VideoLoadAdvanced",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "LoadVideo [n-suite]") return;

		const onAdded = nodeType.prototype.onAdded;
		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onAdded = function () {
			onAdded?.apply(this, arguments);
			const localUrl = this.widgets?.find((widget) => widget.name === "local_url");
			const autoplay = this.widgets?.find((widget) => widget.name === "autoplay");
			const fileInput = document.createElement("input");
			fileInput.type = "file";
			fileInput.accept = "video/mp4,video/webm,image/gif";
			fileInput.hidden = true;
			fileInput.onchange = async () => {
				if (fileInput.files?.length) await uploadFile(fileInput.files[0], true, this);
			};
			document.body.append(fileInput);
			this.__nSuiteVideoFileInput = fileInput;

			const uploadWidget = this.addWidget("button", "choose file to upload", "image", () => fileInput.click());
			uploadWidget.serialize = false;
			ExtendedComfyWidgets.VIDEO(
				this,
				"videoWidget",
				["STRING"],
				localUrl?.value ?? "",
				app,
				"input",
				autoplay?.value ?? true,
			);
		};
		nodeType.prototype.onRemoved = function () {
			this.__nSuiteVideoFileInput?.remove();
			delete this.__nSuiteVideoFileInput;
			onRemoved?.apply(this, arguments);
		};
		nodeType.prototype.onDragOver = function (event) {
			return [...(event.dataTransfer?.items ?? [])].some((item) => item.kind === "file");
		};
		nodeType.prototype.onDragDrop = function (event) {
			let handled = false;
			for (const file of event.dataTransfer?.files ?? []) {
				if (!VIDEO_TYPES.has(file.type)) continue;
				uploadFile(file, !handled, this);
				handled = true;
			}
			return handled;
		};
		nodeType.prototype.pasteFile = function (file) {
			if (!VIDEO_TYPES.has(file.type)) return false;
			uploadFile(file, true, this, true);
			return true;
		};
	},
});
