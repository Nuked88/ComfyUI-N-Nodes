//extended_widgets.js
import { api } from "/scripts/api.js"
import { ComfyWidgets } from "/scripts/widgets.js";

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

function addVideo(node, name,src, app) {
	console.log(src)
	
	const MIN_SIZE = 50;
	function computeSize(size) {
		try{
	
		if (node.widgets[0].last_y == null) return;

		let y = node.widgets[0].last_y;
		let freeSpace = size[1] - y;

		// Compute the height of all non customvideo widgets
		let widgetHeight = 0;
		const multi = [];
		for (let i = 0; i < node.widgets.length; i++) {
			const w = node.widgets[i];
			if (w.type === "customvideo") {
				multi.push(w);
			} else {
				if (w.computeSize) {
					widgetHeight += w.computeSize()[1] + 4;
				} else {
					widgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
				}
			}
		}
	
		// See how large each text input can be
		freeSpace -= widgetHeight;
		freeSpace /= multi.length + (!!node.imgs?.length);

		if (freeSpace < MIN_SIZE) {
			// There isnt enough space for all the widgets, increase the size of the node
			freeSpace = MIN_SIZE;
			node.size[1] = y + widgetHeight + freeSpace * (multi.length + (!!node.imgs?.length));
			node.graph.setDirtyCanvas(true);
		}

		// Position each of the widgets
		for (const w of node.widgets) {
			w.y = y;
			if (w.type === "customvideo") {
				y += freeSpace;
				w.computedHeight = freeSpace - multi.length*4;
			} else if (w.computeSize) {
				y += w.computeSize()[1] + 4;
			} else {
				y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
			}
		}

		node.inputHeight = freeSpace;
	}catch(e){
		
	}
	}
	const widget = {
		type: "customvideo",
		name,
		get value() {
			return this.inputEl.value;
		},
		set value(x) {
			this.inputEl.value = x;
		},
		draw: function (ctx, _, widgetWidth, y, widgetHeight) {
			if (!this.parent.inputHeight) {
				// If we are initially offscreen when created we wont have received a resize event
				// Calculate it here instead
				node.setSizeForImage?.();
				
			}
			const visible = app.canvas.ds.scale > 0.5 && this.type === "customvideo";
			const margin = 10;
			const elRect = ctx.canvas.getBoundingClientRect();
			const transform = new DOMMatrix()
				.scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
				.multiplySelf(ctx.getTransform())
				.translateSelf(margin, margin + y);

			const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
			Object.assign(this.inputEl.style, {
				transformOrigin: "0 0",
				transform: scale,
				left: `${transform.a + transform.e}px`,
				top: `${transform.d + transform.f}px`,
				width: `${widgetWidth - (margin * 2)}px`,
				height: `${this.parent.inputHeight - (margin * 2)}px`,
				position: "absolute",
				background: (!node.color)?'':node.color,
				color: (!node.color)?'':'white',
				zIndex: app.graph._nodes.indexOf(node),
			});
			this.inputEl.hidden = !visible;
		},
	};
	let type_file="mp4";
	const regex = /\.gif&type/;

	if (regex.test(src)) {
		type_file="gif";
	}

	widget.inputEl = document.createElement("div");
		Object.assign(widget.inputEl, {
			id: "videoContainer",
			width: 400,
			height: 300
		})


	if (type_file=="gif"){
		
		let img_element = document.createElement("img");
		Object.assign(img_element, {
			id:"mediaContainer",
			src: src,
			style: "width: 100%; height: 100%;",
			type : "image/gif"
		})

		widget.inputEl.appendChild(img_element);
	}
	else{
		let video_element = document.createElement("video");
			// Set the video attributes
		Object.assign(video_element, {
			id:"mediaContainer",
			controls: true,
			src: src,
			poster: "",
			style: "width: 100%; height: 100%;",
			loop: true,
			muted: true,
			autoplay:true,
			type : "video/mp4"
			
		});
		widget.inputEl.appendChild(video_element);
	}


	


	
	// Add video element to the body
	document.body.appendChild(widget.inputEl);



	widget.parent = node;
	//document.body.appendChild(widget.inputEl);

	node.addCustomWidget(widget);

	app.canvas.onDrawBackground = function () {
		// Draw node isnt fired once the node is off the screen
		// if it goes off screen quickly, the input may not be removed
		// this shifts it off screen so it can be moved back if the node is visible.
		for (let n in app.graph._nodes) {
			n = graph._nodes[n];
			for (let w in n.widgets) {
				let wid = n.widgets[w];
				if (Object.hasOwn(wid, "inputEl")) {
					wid.inputEl.style.left = -8000 + "px";
					wid.inputEl.style.position = "absolute";
				}
			}
		}
	};

	node.onRemoved = function () {
		// When removing this node we need to remove the input from the DOM
		for (let y in this.widgets) {
			if (this.widgets[y].inputEl) {
				this.widgets[y].inputEl.remove();
			}
		}
	};

	widget.onRemove = () => {
		widget.inputEl?.remove();

		// Restore original size handler if we are the last
		if (!--node[MultilineSymbol]) {
			node.onResize = node[MultilineResizeSymbol];
			delete node[MultilineSymbol];
			delete node[MultilineResizeSymbol];
		}
	};

	if (node[MultilineSymbol]) {
		node[MultilineSymbol]++;
	} else {
		node[MultilineSymbol] = 1;
		const onResize = (node[MultilineResizeSymbol] = node.onResize);

		node.onResize = function (size) {
	
			computeSize(size);
			// Call original resizer handler
			if (onResize) {
				onResize.apply(this, arguments);
			}
		};
	}

	return { minWidth: 400, minHeight: 200, widget };
}


export function showVideoInput(name,node) {
	const videoWidget = node.widgets.find((w) => w.name === "videoWidget");
	const temp_web_url = node.widgets.find((w) => w.name === "local_url");
	///const videoContainer = videoWidget.inputEl.widgets.find((w) => w.name === "videoWidget");
	
	let folder_separator = name.lastIndexOf("/");
	let subfolder = "n-suite";
	if (folder_separator > -1) {
		subfolder = name.substring(0, folder_separator);
		name = name.substring(folder_separator + 1);
	}

	let url_video = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}`);
	

	const regex = /\.gif&type/;

	let prev_format = "mp4"
	if (document.getElementById("mediaContainer").tagName=="IMG"){
		prev_format="gif";
	} 
	let current_format = "mp4"
	if (regex.test(url_video)){
		current_format="gif";
	}

	if (prev_format == current_format) { 
		//update
		
		console.log(videoWidget.inputEl)//..getElementById("mediaContainer")
		//videoWidget.inputEl.children[1].src = url_video
	}
	else{
		let newElement;
		if (current_format=="gif"){
			
			newElement = document.createElement("img");
				Object.assign(newElement, {
					id:"mediaContainer",
					src: url_video,
					style: "width: 100%; height: 100%;",
					type : "image/gif"
				})
			}
			else{
				newElement = document.createElement("video");
					// Set the video attributes
				Object.assign(newElement, {
					id:"mediaContainer",
					controls: true,
					src: url_video,
					poster: "",
					style: "width: 100%; height: 100%;",
					loop: true,
					muted: true,
					autoplay:true,
					type : "video/mp4"
					
				});
			}

			let newEl= document.createElement("div");
		Object.assign(newEl, {
			id: "videoContainer",
			width: 400,
			height: 300
		})

		newEl.appendChild(newElement);
		
			videoWidget.lastChilds = newEl;
			console.log(videoWidget)
			console.log("ssssssssss")
			//document.getElementById("videoContainer")
			//videoWidget.inputEl.children[0].replaceChild(newElement,videoWidget.inputEl.children[1]);
		
		
		


	}
	
	
	temp_web_url.value = url_video
}

export function showVideoOutput(name,node) {
	const videoWidget = node.widgets.find((w) => w.name === "videoOutWidget");
	console.log(name)
	
	
	let folder_separator = name.lastIndexOf("/");
	let subfolder = "videos";
	if (folder_separator > -1) {
		subfolder = name.substring(0, folder_separator);
		name = name.substring(folder_separator + 1);
	}


	let url_video = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=output&subfolder=${subfolder}${app.getPreviewFormatParam()}`);
	videoWidget.inputEl.src = url_video

	return url_video;
}



export const ExtendedComfyWidgets = {
    ...ComfyWidgets, // Copy all the functions from ComfyWidgets
	
	VIDEO(node, inputName, inputData, src, app,type="input") {
	try {	
	
		const videoWidget = node.widgets.find((w) => w.name === "video");
		const defaultVal = "";
		let res;
		res = addVideo(node, inputName, src, app);
		
		if (type == "input"){

			const cb = node.callback;
			videoWidget.callback = function () {
				showVideoInput(videoWidget.value, node);
				if (cb) {
					return cb.apply(this, arguments);
				}
			};
		}


		if (node.type =="VideoLoader"){
			// do this only on VideoLoad node!
			let uploadWidget;
			const fileInput = document.createElement("input");
			Object.assign(fileInput, {
				type: "file",
				accept: "video/mp4,image/gif",
				style: "display: none",
				onchange: async () => {
					if (fileInput.files.length) {
						await uploadFile(fileInput.files[0], true,node);
					}
				},
			});
			document.body.append(fileInput);
			// Create the button widget for selecting the files
			uploadWidget = node.addWidget("button", "choose file to upload", "image", () => {
				fileInput.click();
			});
			uploadWidget.serialize = false;

	}

		return res;	
	}
	catch (error) {

		console.error("Errore in extended_widgets.js:", error);
		throw error; 
	
	}

},


};
