import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js"
import { ExtendedComfyWidgets,showVideoInput } from "./extended_widgets.js";
const MultilineSymbol = Symbol();
const MultilineResizeSymbol = Symbol();


async function uploadFile(file, updateNode, node, pasted = false) {
	const videoWidget = node.widgets.find((w) => w.name === "video");
	

	try {
		// Wrap file in formdata so it includes filename
		const body = new FormData();
		body.append("image", file);
		if (pasted) {
			body.append("subfolder", "pasted");
		}
		else {
			body.append("subfolder", "n-suite");
		}
	
		const resp = await api.fetchApi("/upload/image", {
			method: "POST",
			body,
		});

		if (resp.status === 200) {
			const data = await resp.json();
			// Add the file to the dropdown list and update the widget value
			let path = data.name;
			

			if (!videoWidget.options.values.includes(path)) {
				videoWidget.options.values.push(path);
			}
			
			if (updateNode) {
		
				videoWidget.value = path;
				if (data.subfolder) path = data.subfolder + "/" + path;
				showVideoInput(path,node);
				
			}
		} else {
			alert(resp.status + " - " + resp.statusText);
		}
	} catch (error) {
		alert(error);
	}
}




let uploadWidget = "";
app.registerExtension({
	name: "Comfy.VideoLoad",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		const onAdded = nodeType.prototype.onAdded;
		if (nodeData.name === "VideoLoader") {
		nodeType.prototype.onAdded = function () {
			onAdded?.apply(this, arguments);
			const temp_web_url = this.widgets.find((w) => w.name === "local_url");
		
		
		setTimeout(() => {
			ExtendedComfyWidgets["VIDEO"](this, "videoWidget", ["STRING"], temp_web_url.value, app);
		}, 100); 
		
		
		}
	

			nodeType.prototype.onDragOver = function (e) {
				if (e.dataTransfer && e.dataTransfer.items) {
					const image = [...e.dataTransfer.items].find((f) => f.kind === "file");
					return !!image;
				}
	
				return false;
			};
	
			// On drop upload files
			nodeType.prototype.onDragDrop = function (e) {
				console.log("onDragDrop called");
				let handled = false;
				for (const file of e.dataTransfer.files) {
					if (file.type.startsWith("video/mp4") || file.type.startsWith("image/gif")) {
						
						const filePath = file.path || (file.webkitRelativePath || '').split('/').slice(1).join('/'); 


						uploadFile(file, !handled,this ); // Dont await these, any order is fine, only update on first one

						handled = true;
					}
				}
	
				return handled;
			};
	
			nodeType.prototype.pasteFile = function(file) {
				if (file.type.startsWith("image/")) {
					const is_pasted = (file.name === "image.png") &&
									  (file.lastModified - Date.now() < 2000);
					//uploadFile(file, true, is_pasted);

					return true;
				}
				return false;
			}


		};
		
	},
});
