import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";

function buildViewUrl(name, type, defaultSubfolder) {
	const separator = name.lastIndexOf("/");
	const subfolder = separator >= 0 ? name.slice(0, separator) : defaultSubfolder;
	const filename = separator >= 0 ? name.slice(separator + 1) : name;
	const params = new URLSearchParams({ filename, type, subfolder });
	return api.apiURL(`/view?${params.toString()}`);
}

function updateVideoWidget(node, widgetName, url) {
	const widget = node.widgets?.find((item) => item.name === widgetName);
	if (!widget?.element) return;
	widget.element.src = url;
}

function addVideo(node, name, src, autoplayValue) {
	const video = document.createElement("video");
	video.controls = true;
	video.loop = true;
	video.muted = true;
	video.autoplay = autoplayValue;
	video.playsInline = true;
	video.src = src || "";
	video.style.width = "100%";
	video.style.height = "100%";
	video.style.objectFit = "contain";

	const widget = node.addDOMWidget(name, "video", video, {
		hideOnZoom: false,
		getMinHeight: () => 200,
		getHeight: () => 240,
	});
	widget.serialize = false;
	widget.options.serialize = false;
	return { minWidth: 400, minHeight: 200, widget };
}

export function showVideoInput(name, node) {
	const url = buildViewUrl(name, "input", "n-suite");
	updateVideoWidget(node, "videoWidget", url);
	const localUrl = node.widgets?.find((item) => item.name === "local_url");
	if (localUrl) localUrl.value = url;
	return url;
}

export function showVideoOutput(name, node) {
	const url = buildViewUrl(name, "output", "n-suite/videos");
	updateVideoWidget(node, "videoOutWidget", url);
	return url;
}

export const ExtendedComfyWidgets = {
	...ComfyWidgets,
	VIDEO(node, inputName, _inputData, src, _app, type = "input", autoplayValue = true) {
		const result = addVideo(node, inputName, src, autoplayValue);
		if (type !== "input") return result;

		const video = node.widgets?.find((item) => item.name === "video");
		const autoplay = node.widgets?.find((item) => item.name === "autoplay");
		if (video) {
			const callback = video.callback;
			video.callback = function () {
				showVideoInput(video.value, node);
				return callback?.apply(this, arguments);
			};
		}
		if (autoplay) {
			const callback = autoplay.callback;
			autoplay.callback = function () {
				const preview = node.widgets?.find((item) => item.name === "videoWidget");
				if (preview?.element) preview.element.autoplay = autoplay.value;
				if (video?.value) showVideoInput(video.value, node);
				return callback?.apply(this, arguments);
			};
		}
		return result;
	},
};
