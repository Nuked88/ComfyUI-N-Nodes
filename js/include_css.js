function addStylesheet(url) {
	if (url.endsWith(".js")) {
		url = url.substr(0, url.length - 2) + "css";
	}
	const link = document.createElement("link");
	link.rel = "stylesheet";
	link.href = url.startsWith("http") ? url : getUrl(url);
	document.head.append(link);
}
function getUrl(path, baseUrl) {
	if (baseUrl) {
		return new URL(path, baseUrl).toString();
	} else {
		return new URL("../" + path, import.meta.url).toString();
	}
}

addStylesheet(getUrl("styles.css", import.meta.url));
