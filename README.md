# ConfyUI-N-Nodes
A suite of custom nodes for ConfyUI, for now i just put Integer, string and float variable nodes

# Installation

1. Clone the repository:
`git clone https://github.com/Nuked88/ConfyUI-N-Nodes.git`  
to your ComfyUI `custom_nodes` directory

   ComfyUI will then automatically load all custom scripts and nodes at the start.  


- For uninstallation:
  - Delete the cloned repo in `custom_nodes`

# Update
1. Navigate to the cloned repo e.g. `custom_nodes/ConfyUI-N-Nodes`
2. `git pull`

# Features
Since the primitive node has limitations in links (for example at the time i'm writing you cannot link "start_at_step" and "steps" of another ksampler toghether), I decided to create these simple node-variables to bypass this limitation
The node-variables are:
- Integer
- Float
- String

